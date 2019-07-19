from module.pipeline import Preprocessing as pre
from module.pipeline import FeatureExtraction as fea
from module.pipeline.FeatureSelection import FeaturesSelection, ChanelsSelection, SensorsSelection
from module.pipeline import Classification as cla
import logging

class Evaluation:

    def __init__(self, cli, config):
        self.cli = cli
        self.config = config

    def make_preprocessing(self):
        print("make_preprocessing")

        preprocess = pre.Preprocessing(type=self.config.type, raw=self.config.raw, use_diff=self.config.use_diff, processes=[
            pre.RemoveSensors(only=self.config.sensors),
            pre.SlidingWindow(size=self.config.window_size, step=self.config.window_step),
            pre.RemoveFirstWindows(seconds=self.config.remove_seconds),
        ])
        preprocess.client = self.cli
        dfs = preprocess.execute()
        return dfs


    def make_feature_engineering(self, dfs):
        print("make_feature_engineering")

        ts_features = {}
        for f in self.config.features:
            if f in ["abs_integral", "velocity", "angular_velocity"]:
                continue
            ts_features[f] = None

        featureExtraction = fea.FeatureEngineering(
            ts_features=ts_features,
            configFeatures=self.config.features,
            combine=("q" in self.config.chanels))

        featureExtraction.client = self.cli
        features_df = featureExtraction.execute(dfs)

        # hm clear cluster..

        logging.getLogger('dask_jobqueue.core').setLevel(logging.CRITICAL)
        self.cli.restart()
        logging.getLogger('dask_jobqueue.core').setLevel(logging.WARNING)

        return features_df


    def make_selection(self, features_df):
        print("make_selection")

        df_1 = FeaturesSelection(self.config.features).execute(features_df)
        df_2 = ChanelsSelection(self.config.chanels).execute(df_1)
        df_3 = SensorsSelection(self.config.sensors).execute(df_2)
        return df_3


    def make_classification(self, data):
        print("make_classification")

        classifi = cla.Classification()
        classifi.client = self.cli
        return classifi.execute(data)


