from module.Configuration import Configuration
from module.Evaluation import Evaluation
from dask.distributed import Client
import logging
import math
import pandas as pd


def transform(df, parts=1):
    newdfs = []
    for idx, group in df.groupby(level=0):
        fileSize = len(group)
        partSize = math.ceil(fileSize / parts)
        dfPerGroup = []
        for i in range(parts):
            low = i * partSize
            high = i * partSize + partSize
            dd = group.iloc[low:high, :]
            dd.index = dd.index.astype(str) + '_%d'%i
            dfPerGroup.append(dd)
        newdf = pd.concat(dfPerGroup)
        newdfs.append(newdf)
    result = pd.concat(newdfs)
    result.index.name = 'id'
    return result


def process_index(k):
    return tuple(["_".join(k.split("_")[:-1]), k.split("_")[-1]])


def main():
    conf = Configuration(window_size=20, window_step=5.0, features=['standard_deviation'])
    print(conf)

    c = Client()

    eva = Evaluation(c, conf)

    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    df1 = eva.make_preprocessing()

    df2 = map(lambda df: transform(df, parts=5), df1)
    df2 = list(df2)

    df3 = eva.make_feature_engineering(df2)
    tmpdf = df3.copy()
    tmpdf.index = pd.MultiIndex.from_tuples([process_index(k) for k, v in tmpdf.iterrows()])
    newgrouping = []
    for idx, group in tmpdf.groupby(level=0):
        newrow = []
        for idx2, row in group.iterrows():
            row.index = row.index.astype(str) + '_%s'%idx2[1]
            newrow.append(pd.DataFrame([row.values], columns=list(row.index)))

        newdf = pd.concat(newrow, sort=True, axis=1)
        newdf.index = [idx]
        newgrouping.append(newdf)
    df4 = pd.concat(newgrouping)
    df5 = eva.make_selection(df4)
    results_clf_score, sizes, info_df, importances_df, all_probas = eva.make_classification(df5)


if __name__ == "__main__":
    main()
