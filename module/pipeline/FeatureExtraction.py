from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.distribution import MapDistributor
import pandas as pd
import numpy as np
import skinematics as skin
import logging


def ts_extract(df, features):
    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    # participant = file.split("_")[0]
    # video = file.split("_")[1]
    dd = df.copy()
    dd['id'] = dd.index
    extracted_features = extract_features(dd, column_id="id", column_sort="time", default_fc_parameters = features, distributor=MapDistributor())
    impute(extracted_features)
    return extracted_features


def abs_integral(df):
    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    grouped = df.groupby(by="id")
    rows = []
    cols = None
    newdfs = []
    for name, group in grouped:
        tmin = group.time.min()
        dt = group.time - tmin

        grp = group.loc[:, group.columns.map(lambda column: column != "time")]
        integral = {}
        for col in grp.columns:
            colname = col + "__abs_integral"
            integral[colname] = np.trapz(grp[col].abs(), x=dt/1000)
        dd = pd.DataFrame(integral, index=[name])
        newdfs.append(dd)
    return pd.concat(newdfs)


def velocity(df):
    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    grouped = df.groupby(by="id")
    rows = []
    cols = None
    newdfs = []
    for name, group in grouped:
        dt = group.time.max() - group.time.min()

        grp = group.loc[:, group.columns.map(lambda column: column != "time")]
        velocity = {}
        for col in grp.columns:
            colname = col + "__velocity" # Sensor_x__velocity
            velocity[colname] = grp[col].diff().dropna().abs().sum() / (dt/1000)
        dd = pd.DataFrame(velocity, index=[name])
        newdfs.append(dd)
    return pd.concat(newdfs)


def rotation_velocity(df):
    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    grouped = df.groupby(by="id")
    rows = []
    cols = None
    newdfs = []
    for name, group in grouped: # group per id 'P01_IT_001' etc.
        grp = group.loc[:, group.columns.map(lambda column: column != "time")] # all rows without time column
        sensors = list(set(grp.columns.map(lambda column: column.split("_")[0])))
        qsPerSensor = {}
        for index, row in grp.iterrows():
            for sensor in sensors:
                q = np.array([row['%s_w'%sensor], row['%s_x'%sensor], row['%s_y'%sensor], row['%s_z'%sensor]])
                if sensor not in qsPerSensor:
                    qsPerSensor[sensor] = []
                qsPerSensor[sensor].append(q)
        time = ( (group.time.max() - group.time.min()) / 1000 )
        hz = len(group) / time

        angularVelocity = {}
        for sensor, qs in qsPerSensor.items():
            qs = np.array(qs)
            angVel = skin.quat.calc_angvel(qs, rate=hz, winSize=5, order=2)
            angVel = np.sum(np.diff(angVel, axis=0), axis=0) / time
            angularVelocity["%s_x__angular_velocity"%sensor] = angVel[0]
            angularVelocity["%s_y__angular_velocity"%sensor] = angVel[1]
            angularVelocity["%s_z__angular_velocity"%sensor] = angVel[2]

        dd = pd.DataFrame(angularVelocity, index=[name])
        dd = dd.fillna(0)
        newdfs.append(dd)
    return pd.concat(newdfs)


def combine_feature(df):
    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    # combine Sensor_%s_Feature together to Sensor_q_Feature
    # need a list of Sensors and a list of Features from df.columns
    sensors = set(map(lambda c: c.split("_")[0], df.columns))
    features = set(map(lambda c: c.split("__")[1], df.columns))

    for sensor in sensors:
        for feature in features:
            q = "%s_q__%s" % (sensor, feature)
            x = "%s_x__%s" % (sensor, feature)
            y = "%s_y__%s" % (sensor, feature)
            z = "%s_z__%s" % (sensor, feature)
            w = "%s_w__%s" % (sensor, feature)

            if w in df.columns:
                df[q] = df[x]+df[y]+df[z]+df[w]
            else:
                df[q] = df[x]+df[y]+df[z]
    return df


class FeatureEngineering():

    def __init__(self, ts_features, configFeatures, combine=True):
        self.ts_features = ts_features
        self.configFeatures = configFeatures
        self.combine = combine

        logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    def execute(self, preprocessed_dfs):
        finaldf = self.read_input(preprocessed_dfs)

        if self.combine:
            finaldf = combine_feature(finaldf)

        finaldf = finaldf.sort_index()
        finaldf = finaldf.reindex(sorted(finaldf.columns), axis=1)
        return finaldf

    def read_input(self, preprocessed_dfs):
        tsfs = [self.ts_features for i in range(len(preprocessed_dfs))]

        dfs_return = []
        big_futures = self.client.scatter(preprocessed_dfs)
        futures = self.client.map(ts_extract, big_futures, tsfs)
        results = self.client.gather(futures)
        df_ts = pd.concat(results)
        dfs_return.append(df_ts)

        if "velocity" in self.configFeatures:
            futures = self.client.map(velocity, big_futures)
            results = self.client.gather(futures)
            df_vel = pd.concat(results)
            dfs_return.append(df_vel)

        if "angular_velocity" in self.configFeatures:
            futures = self.client.map(rotation_velocity, big_futures)
            results = self.client.gather(futures)
            df_rot = pd.concat(results, sort=True)
            dfs_return.append(df_rot)

        if "abs_integral" in self.configFeatures:
            futures2 = self.client.map(abs_integral, big_futures)
            results2 = self.client.gather(futures2)
            df_int = pd.concat(results2)
            dfs_return.append(df_int)

        return pd.concat(dfs_return, axis=1)

    def config(self):
        return {"features": self.features}
