from module.Configuration import Configuration
from module.Evaluation import Evaluation
# from module.Cluster import Cluster
from dask.distributed import Client
import logging


def main():

    c = Client()

    # On a SLURM Network, you can call:
    # clu = Cluster()
    # c = clu.cli
    # Check module/Cluster.py for more details


    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    for sensor in ['Pelvis', 'Chest', 'Head', 'RightLowerArm', 'RightUpperArm', 'LeftLowerArm', 'LeftUpperArm']:
        conf = Configuration(window_size=20, window_step=5.0, sensors=[sensor])
        eva = Evaluation(c, conf)
        df1 = eva.make_preprocessing()
        df2 = eva.make_feature_engineering(df1)
        df3 = eva.make_selection(df2)
        results_clf_score, sizes, info_df, importances_df, all_probas = eva.make_classification(df3)

        sensor, results_clf_score["RandomForest"]


    # clu.close()


if __name__ == "__main__":
    main()
