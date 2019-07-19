from module.pipeline import Preprocessing as pre
from module.pipeline import FeatureExtraction as fea
from module.pipeline.FeatureSelection import FeaturesSelection, ChanelsSelection, SensorsSelection
from module.pipeline import Classification as cla
import logging


def make_preprocessing(cli, config, type="orientation", raw="ned", use_diff=False):
    print("make_preprocessing")

    preprocess = pre.Preprocessing(type=type, raw=raw, use_diff=use_diff, processes=[
        pre.RemoveSensors(only=config.sensors),
        pre.SlidingWindow(size=config.window_size, step=config.window_step),
        pre.RemoveFirstWindows(seconds=config.remove_seconds),
    ])
    preprocess.client = cli
    dfs = preprocess.execute()
    return dfs


def make_feature_engineering(cli, config, dfs):
    print("make_feature_engineering")

    featureExtraction = fea.FeatureEngineering(ts_features={
        'mean': None,
        'standard_deviation': None,
        'count_above_mean': None,
        'count_below_mean': None,
        'abs_energy': None
    }, configFeatures=config.features, calc_integral=True, combine=True)
    featureExtraction.client = cli
    features_df = featureExtraction.execute(dfs)

    # hm clear cluster..

    logging.getLogger('dask_jobqueue.core').setLevel(logging.CRITICAL)
    cli.restart()
    logging.getLogger('dask_jobqueue.core').setLevel(logging.WARNING)

    return features_df


def make_selection(cli, config, features_df):
    print("make_selection")

    df_1 = FeaturesSelection(config.features).execute(features_df)
    df_2 = ChanelsSelection(config.chanels).execute(df_1)
    df_3 = SensorsSelection(config.sensors).execute(df_2)
    return df_3


def make_classification(cli, data):
    print("make_classification")

    classifi = cla.Classification()
    classifi.client = cli
    return classifi.execute(data)


