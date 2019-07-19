from module.pipeline.Configuration import Configuration
from dask.distributed import Client
import logging


def main():
    eva = __import__("Evaluation")

    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    cli = Client()

    standard_config = Configuration(window_size=20, window_step=5.0)
    print(standard_config)
    df1 = eva.make_preprocessing(cli, standard_config)
    df2 = eva.make_feature_engineering(cli, standard_config, df1)
    df3 = eva.make_selection(cli, standard_config, df2)
    results_clf_score, sizes, info_df, importances_df, all_probas = eva.make_classification(cli, df3)


if __name__ == "__main__":
    main()
