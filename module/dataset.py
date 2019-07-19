import pandas as pd
from sklearn.model_selection import train_test_split
from module import labeling


def _video_of(name, nrs):
    file = "./data/metadata/questionnaire.csv"
    metadf = pd.read_csv(file, sep=";", index_col="name")
    return list(map(lambda nr: metadf.loc[name]["video%d"%nr], nrs))

def select_participants(df, participants):
	return df.loc[df.index.map(lambda x: any(arg+"_" in x for arg in participants))]

def select_sensors(df, sensors):
	return df.loc[:, df.columns.map(lambda x: any(arg+"_" in x for arg in sensors))]

def select_video(df, videos=[1, 2, 3, 4]):
    return df.loc[df.index.map(lambda idx: any(video in idx for video in _video_of(idx.split("_")[0], videos) ))]
    
def select_sensor_value(df, values=["x", "y", "z", "w"]):
    return df.loc[:, df.columns.map(lambda x: any("_q"+value+"__" in x for value in values))]

def select_features(df, features):
    return df.loc[:, df.columns.map(lambda x: any("__"+feature in x for feature in features))]
    
def select(df, participants=None, sensors=None, videos=None):
    tmp = df
    if participants is not None:
        tmp = tmp.loc[tmp.index.map(lambda x: any(arg+"_" in x for arg in participants))]
    if sensors is not None:
        tmp = tmp.loc[:, tmp.columns.map(lambda x: any(arg+"_" in x for arg in sensors))]
    if videos is not None:
        tmp = tmp.loc[tmp.index.map(lambda idx: any(video in idx for video in _video_of(idx.split("_")[0], videos) ))]
    return tmp

def split(df, 
          participants=["P%02d" % x for x in range(1, 22)],
          sensors=["Head", "Chest", "LeftUpperArm", "LeftLowerArm", 
                   "RightUpperArm", "RightLowerArm", "Pelvis", "LeftUpperLeg", 
                   "LeftLowerLeg", "RightUpperLeg", "RightLowerLeg"],
          videos=[1, 2, 3, 4],
          participant_test_size=None,
          test_size=None,
          label=labeling.SimpleConfig(),
          random_state=None):
    

    if participant_test_size is None and test_size is not None:
        df = select(df, participants=participants, sensors=sensors, videos=videos)
        y = labeling.get_label(df, label)
        return train_test_split(df, y, test_size=test_size, random_state=random_state)
    
    elif participant_test_size is not None and test_size is None:
        p_train, p_test = train_test_split(participants,test_size=participant_test_size, random_state=random_state) 

        df_train = select(df, participants=p_train, sensors=sensors, videos=videos)
        df_test = select(df, participants=p_test, sensors=sensors, videos=videos)

        y_train = labeling.get_label(df_train, label)
        y_test = labeling.get_label(df_test, label)

        return df_train, df_test, y_train, y_test
    else:
        return None
        
