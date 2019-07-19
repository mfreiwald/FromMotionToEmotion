from module.Configuration import Configuration
from module.Evaluation import Evaluation
from dask.distributed import Client
import logging


def main():
    conf = Configuration(window_size=20, window_step=5.0)
    print(conf)

    c = Client()
    eva = Evaluation(c, conf)

    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    df1 = eva.make_preprocessing()
    df2 = eva.make_feature_engineering(df1)
    df3 = eva.make_selection(df2)
    results_clf_score, sizes, info_df, importances_df, all_probas = eva.make_classification(df3)


if __name__ == "__main__":
    main()
