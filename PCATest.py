from module.pipeline.Configuration import Configuration
from dask.distributed import Client
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from module.Evaluation import Evaluation
# from module.Cluster import Cluster


def main():

    c = Client()

    # On a SLURM Network, you can call:
    # clu = Cluster()
    # c = clu.cli
    # Check module/Cluster.py for more details

    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    standard_config = Configuration(window_size=20, window_step=5.0, remove_seconds=0)
    print(standard_config)
    eva = Evaluation(c, standard_config)

    df1 = eva.make_preprocessing()
    df2 = eva.make_feature_engineering(df1)
    df3 = eva.make_selection(df2)

    for pcaNr in [2, 5, 25, 66, 100, 150, 200]:
        df = df3.copy()
        pcaCols = ["PC%03d"%d for d in range(pcaNr)]
        x = df.values
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=pcaNr)
        principalComponents = pca.fit_transform(x)
        df4 = pd.DataFrame(data = principalComponents, columns = pcaCols, index=df.index)
        'PCA:', pcaNr
        results_clf_score, sizes, info_df, importances_df, all_probas = eva.make_classification(df4)

    # clu.close()


if __name__ == "__main__":
    main()
