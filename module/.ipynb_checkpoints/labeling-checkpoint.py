import pandas as pd
import numpy as np


class LabelConfig:
    relax = "relax" # 1
    tense = "tense" # 0

    def __init__(self, relax = "relax", tense = "tense"):
        self.relax = relax
        self.tense = tense

    def label(self, id: str):
        return None


class SimpleConfig(LabelConfig):

    def __init__(self, relax = "relax", tense = "tense"):
        super().__init__(relax=relax, tense=tense)

    def label(self, id: str):
        if "Nun" in id or "IT" in id:
            return self.tense
        elif "World" in id or "Relax" in id:
            return self.relax
        else:
            return None


class RankingConfig(LabelConfig):

    rankingFile = "./data/metadata/videos.csv"

    def __init__(self, relax = "relax", tense = "tense"):
        super().__init__(relax=relax, tense=tense)
        self.df = pd.read_csv(self.rankingFile, sep=";", index_col=["name", "video"])
        pass

    def read_rank(self, id):
        participant = id.split("_")[0]
        video = id.split("_")[1]
        return int(self.df.loc[participant, video].ranking)

    def label(self, id: str):
        rank = self.read_rank(id)
        if rank < 3:
            return self.relax
        elif rank > 3:
            return self.tense
        else:
            if "Nun" in id or "IT" in id:
                return self.tense
            elif "World" in id or "Relax" in id:
                return self.relax
            else:
                return None

            
class OnlineConfig(LabelConfig):
    rankingFile = "./data/metadata/online_results.csv"

    def __init__(self, relax = "relax", tense = "tense"):
        super().__init__(relax=relax, tense=tense)
        self.df = pd.read_csv(self.rankingFile, sep=";")
    
    def label(self, id: str):
        participant = id.split("_")[0]
        video = id.split("_")[1]
        return self.df[(self.df.Participant == participant) & (self.df.Video == video)].Label.max()
            
        
class ExpertConfig(LabelConfig):
    rankingFile = "./data/metadata/expert.csv"

    def __init__(self, relax = "relax", tense = "tense"):
        super().__init__(relax=relax, tense=tense)
        self.df = pd.read_csv(self.rankingFile, sep=";")
    
    def label(self, id: str):
        participant = id.split("_")[0]
        video = id.split("_")[1]
        return self.df[(self.df.Participant == participant) & (self.df.Video == video)].Label.max()
    
    
class RankingThresholdConfig(LabelConfig):

    rankingFile = "./data/metadata/videos.csv"

    def __init__(self, threshold = 2.5, relax = "relax", tense = "tense"):
        super().__init__(relax=relax, tense=tense)
        self.threshold = threshold
        self.df = pd.read_csv(self.rankingFile, sep=";", index_col=["name", "video"])
        pass

    def read_rank(self, id):
        participant = id.split("_")[0]
        video = id.split("_")[1]
        return int(self.df.loc[participant, video].ranking)

    def label(self, id: str):
        rank = self.read_rank(id)
        if rank <= self.threshold:
            return self.relax
        else:
            return self.tense

        
class RegressionConfig(LabelConfig):

    rankingFile = "./data/metadata/videos.csv"

    def __init__(self, relax = "relax", tense = "tense"):
        super().__init__(relax=relax, tense=tense)
        self.df = pd.read_csv(self.rankingFile, sep=";", index_col=["name", "video"])
        pass

    def read_rank(self, id):
        participant = id.split("_")[0]
        video = id.split("_")[1]
        return int(self.df.loc[participant, video].ranking)

    def label(self, id: str):
        rank = self.read_rank(id)
        return rank


def get_label(id, config: LabelConfig):
    if type(id) is str:
        return config.label(id)
    elif type(id) is pd.DataFrame:
        return pd.Series(map(config.label, id.index), index=id.index)


def stats(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
    
    
#print("Test LabelConfig")

#df = pd.read_csv("./data/featuresAll/local_10_1/P01_Nun.csv", sep=";", index_col="id")
#id = "P16_Relax_5"

#df = pd.read_csv("/big/f/freiwald/local_10_1.csv", index_col="id")
#config = RankingConfig()
#label = get_label(df, config)
#print(label)
