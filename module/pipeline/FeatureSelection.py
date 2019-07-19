import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class FeaturesSelection():
    def __init__(self, features):
        self.features = features

    def execute(self, df):
        return df.loc[:, df.columns.map(lambda column: any("__"+feature in column for feature in self.features))]


class ChanelsSelection():
    def __init__(self, chanels):
        self.chanels = chanels

    def execute(self, df):
        return df.loc[:, df.columns.map(lambda column: any("_"+chanel+"__" in column for chanel in self.chanels))]


class SensorsSelection():
    def __init__(self, sensors):
        self.sensors = sensors

    def execute(self, df):
        return df.loc[:, df.columns.map(lambda column: any(sensor+"_" in column for sensor in self.sensors))]


class PCASelection():
    def __init__(self, number, scale=False):
        self.number = number
        self.scale = scale

    def execute(self, df):
        columns = ['PC_%03d'%i for i in range(1,self.number+1)]
        x = df.loc[:, :].values
        if self.scale:
            x = StandardScaler().fit_transform(x)
        # principalComponents = KernelPCA(n_components=self.number, kernel='sigmoid').fit_transform(x)
        principalComponents = KernelPCA(n_components=self.number).fit_transform(x)
        return pd.DataFrame(data = principalComponents, columns = columns, index=df.index)
