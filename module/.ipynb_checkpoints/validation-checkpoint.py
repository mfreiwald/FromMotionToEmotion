import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
from sklearn import metrics
from module import dataset, labeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def _cv_fold(data, participants, sensors, features, videos_train, videos_test, labelConfig, random_state, train_index, test_index):
    train_p = participants[train_index]
    test_p = participants[test_index]

    train_data = dataset.select_participants(data, train_p)
    test_data = dataset.select_participants(data, test_p)

    train_data = dataset.select_sensors(train_data, sensors)
    test_data = dataset.select_sensors(test_data, sensors)
    
    train_data = dataset.select_features(train_data, features)
    test_data = dataset.select_features(test_data, features)

    train_data = dataset.select_video(train_data, videos_train)
    test_data = dataset.select_video(test_data, videos_test)

    train_x = shuffle(train_data, random_state=random_state)
    test_x = shuffle(test_data, random_state=random_state)

    train_y = labeling.get_label(train_x, labelConfig)
    test_y = labeling.get_label(test_x, labelConfig)
        
    return (train_x, test_x, train_y, test_y, test_p[0])

        
def cv_fold(
    slidingWindow, 
    participants=np.array(["P%02.d"%d for d in range(1,21) if d != 11]), 
    sensors=["Head", "Chest", "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm", "Pelvis"], 
    videos_train = [1, 2, 3, 4],
    videos_test = [1, 2, 3, 4],
    features = ["sum_values", "median", "mean", "standard_deviation", "variance", "maximum", "minimum", "count_above_mean", "count_below_mean", "longest_strike_below_mean", "longest_strike_above_mean", "cid_ce", "number_peaks", "autocorrelation", "quantile"], 
    labelConfig=labeling.SimpleConfig(relax="relax", tense="tense"),
    scaler = None,
    random_state = None):
    
    executer = ThreadPoolExecutor(max_workers=8)

    result = {}
    
    file = "/big/f/freiwald/featuresets/local_%d_%.1f.csv" % (10, slidingWindow)
    data = pd.read_csv(file, index_col="id")

    all_y = labeling.get_label(data, labelConfig)
    
    if scaler is not None:
        x_scaled = scaler.fit_transform(data)
        data = pd.DataFrame(x_scaled, index=data.index, columns=data.columns)

    result["labelCount"] = all_y.value_counts()

    loo = LeaveOneOut()    
    loo.get_n_splits(participants)

    splits = []
    futures = []
    for train_index, test_index in loo.split(participants):
        future = executer.submit(_cv_fold, data, participants, sensors, features, videos_train, videos_test, labelConfig, random_state, train_index, test_index)
        futures.append(future)
    
    for future in futures:
        splits.append(future.result())
        
    result["splits"] = splits
    return result


def cross_validation(folds,
                    clf):
    
    result = {}
    
    result["participants"] = {}
    result["Scores"] = {}
    result["ScoreStd"] = {}
    
    
    di = {
        "Head": [],
        "Chest": [],
        "LeftUpperArm": [],
        "LeftLowerArm": [],
        "RightUpperArm": [],
        "RightLowerArm": [],
        "Pelvis": [],
    }
    
    for train_x, test_x, train_y, test_y, participant in folds["splits"]:
        
        model = clf.fit(train_x, train_y)
        
        features = pd.Series(model.feature_importances_, index=train_x.columns)

        d = {}
        for idx, f in enumerate(features):
            s = list(features.index)[idx].split("_")[0]
            if s in d:
                d[s] += f
            else:
                d[s] = 0
            
        for k,v in d.items():
            di[k].append(v)
           
        # print(participant, d) # features.sort_values(ascending=False)[:10])
        
        result["participants"][participant] = {}
        
        # Split Test Set into Videos
        videos_world_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "World")]
        videos_relax_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "Relax")]
        videos_it_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "IT")]
        videos_nun_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "Nun")]
        
        videos_world_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "World")]
        videos_relax_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "Relax")]
        videos_it_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "IT")]
        videos_nun_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "Nun")]
        
        
        result["participants"][participant]["Scores"] = {}
        result["participants"][participant]["Matrix"] = {}

        if len(videos_world_x) > 0:
            predict_world_y = model.predict(videos_world_x)
            result["participants"][participant]["Scores"]["World"] = model.score(videos_world_x, videos_world_y)
            result["participants"][participant]["Matrix"]["World"] = metrics.confusion_matrix(videos_world_y, predict_world_y)
       #else:
       #     result["participants"][participant]["Scores"]["World"] = 0
            
            
        if len(videos_relax_x) > 0:
            predict_relax_y = model.predict(videos_relax_x)
            result["participants"][participant]["Scores"]["Relax"] = model.score(videos_relax_x, videos_relax_y)
            result["participants"][participant]["Matrix"]["Relax"] = metrics.confusion_matrix(videos_relax_y, predict_relax_y)
       # else:
       #     result["participants"][participant]["Scores"]["Relax"] = 0
            
        
        if len(videos_it_x) > 0:
            predict_it_y = model.predict(videos_it_x)
            result["participants"][participant]["Scores"]["IT"] = model.score(videos_it_x, videos_it_y)
            result["participants"][participant]["Matrix"]["IT"] = metrics.confusion_matrix(videos_it_y, predict_it_y)
        #else:
       #     result["participants"][participant]["Scores"]["IT"] = 0
        
        if len(videos_nun_x) > 0:
            predict_nun_y = model.predict(videos_nun_x)
            result["participants"][participant]["Scores"]["Nun"] = model.score(videos_nun_x, videos_nun_y)
            result["participants"][participant]["Matrix"]["Nun"] = metrics.confusion_matrix(videos_nun_y, predict_nun_y)
        #else:
            #result["participants"][participant]["Scores"]["Nun"] = 0
        
        predict_y = model.predict(test_x)
        result["participants"][participant]["Scores"]["All"] = model.score(test_x, test_y)
        result["participants"][participant]["Matrix"]["All"] = metrics.confusion_matrix(test_y, predict_y)

        
    if "World" in result["participants"]["P01"]["Scores"]:
        result["Scores"]["World"] = np.array(list(map(lambda v: v["Scores"]["World"], result["participants"].values()))).mean()
        result["ScoreStd"]["World"] = np.array(list(map(lambda v: v["Scores"]["World"], result["participants"].values()))).std()
    
    if "Relax" in result["participants"]["P01"]["Scores"]:
        result["Scores"]["Relax"] = np.array(list(map(lambda v: v["Scores"]["Relax"], result["participants"].values()))).mean()
        result["ScoreStd"]["Relax"] = np.array(list(map(lambda v: v["Scores"]["Relax"], result["participants"].values()))).std()
    
    if "IT" in result["participants"]["P01"]["Scores"]:
        result["Scores"]["IT"] = np.array(list(map(lambda v: v["Scores"]["IT"], result["participants"].values()))).mean()
        result["ScoreStd"]["IT"] = np.array(list(map(lambda v: v["Scores"]["IT"], result["participants"].values()))).std()
    
    if "Nun" in result["participants"]["P01"]["Scores"]:
        result["Scores"]["Nun"] = np.array(list(map(lambda v: v["Scores"]["Nun"], result["participants"].values()))).mean()
        result["ScoreStd"]["Nun"] = np.array(list(map(lambda v: v["Scores"]["Nun"], result["participants"].values()))).std()

    result["Scores"]["All"] = np.array(list(map(lambda v: v["Scores"]["All"], result["participants"].values()))).mean()
    result["ScoreStd"]["All"] = np.array(list(map(lambda v: v["Scores"]["All"], result["participants"].values()))).std()
    
    for k,v in di.items():
        print(k, np.mean(v))
    
    return result


def evaluate(estimators,
            slidingWindow,
            participants=np.array(["P%02.d"%d for d in range(1,21) if d != 11]), 
            sensors=["Head", "Chest", "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm", "Pelvis"], 
            videos_train = [1, 2, 3, 4],
            videos_test = [1, 2, 3, 4],
            features = ["sum_values", "median", "mean", "standard_deviation", "variance", "maximum", "minimum", "count_above_mean", "count_below_mean", "longest_strike_below_mean", "longest_strike_above_mean", "cid_ce", "number_peaks", "autocorrelation", "quantile"], 
            labelConfig=labeling.SimpleConfig(relax="relax", tense="tense"),
            scaler = None,
            random_state = None):
    
    folds = cv_fold(slidingWindow = slidingWindow,
                   participants = participants,
                   sensors = sensors,
                   videos_train = videos_train,
                   videos_test = videos_test,
                   features = features,
                   labelConfig = labelConfig,
                   scaler = scaler,
                   random_state = random_state)
    
    results = {}
    for name, clf in estimators.items():
        cv = cross_validation(folds, clf)
        results[name] = cv
    
    return results


def plot_scores(cvresult):
    legends = []
    #for name, cvresult in result.items():
    scores = list(map(lambda p: p["Scores"]["All"], cvresult["participants"].values()))
    plt.plot(cvresult["participants"].keys(), scores)
    legends.append("%s - %.3f" % ("", np.mean(scores) ))
    plt.legend(legends)

    
def plot_video_scores(cvresult):
    legends = []
    
    if "World" in cvresult["participants"]["P01"]["Scores"]:
        world = list(map(lambda p: p["Scores"]["World"], cvresult["participants"].values()))
        plt.plot(cvresult["participants"].keys(), world)
        legends.append("%s - %.3f" % ("World", np.mean(world) ))
        
        
    if "Relax" in cvresult["participants"]["P01"]["Scores"]:
        relax = list(map(lambda p: p["Scores"]["Relax"], cvresult["participants"].values()))
        plt.plot(cvresult["participants"].keys(), relax)
        legends.append("%s - %.3f" % ("Relax", np.mean(relax) ))
        
    if "IT" in cvresult["participants"]["P01"]["Scores"]:
        it = list(map(lambda p: p["Scores"]["IT"], cvresult["participants"].values()))
        plt.plot(cvresult["participants"].keys(), it)
        legends.append("%s - %.3f" % ("IT", np.mean(it) ))

    if "Nun" in cvresult["participants"]["P01"]["Scores"]:
        nun = list(map(lambda p: p["Scores"]["Nun"], cvresult["participants"].values()))
        plt.plot(cvresult["participants"].keys(), nun)
        legends.append("%s - %.3f" % ("Nun", np.mean(nun) ))
    
    plt.legend(legends)

def score(result):
    scores = {}
    for name, cvresult in result.items():
        scor = list(map(lambda x: x["score"], cvresult.values()))
        scores[name] = np.mean(scor)
    return scores


def scores(result):
    scores = {}
    for name, cvresult in result.items():
        scor = list(map(lambda x: x["score"], cvresult.values()))
        scores[name] = np.array(scor)
    return scores


def real_score(result):
    scores = {}
    for name, cvresult in result.items():
        scores[name] = {}
        for participant, values in cvresult.items():
            test_y = values["test_y"]
            predict_y = values["predict_y"]

            videosScore = {}
            '''
                "Relax": {"relax":0, "tense": 0},
                "World": {"relax":0, "tense": 0},
                "IT": {"relax":0, "tense": 0},
                "Nun": {"relax":0, "tense": 0},
            }'''

            for idx, ty in enumerate(test_y):
                video = test_y.index[idx].split("_")[1]
                video = video + "(%s)" % ty
                py = predict_y[idx]
                print(video, py)


            #scores[name][participant] = videosScore
    return scores


def participant_validation_fold(
        slidingWindow, 
        participants=np.array(["P%02.d"%d for d in range(1,21) if d != 11]), 
        sensors=["Head", "Chest", "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm", "Pelvis"], 
        videos_train = [1, 2, 3, 4],
        videos_test = [1, 2, 3, 4],
        features = ["sum_values", "median", "mean", "standard_deviation", "variance", "maximum", "minimum", "count_above_mean", "count_below_mean", "longest_strike_below_mean", "longest_strike_above_mean", "cid_ce", "number_peaks", "autocorrelation", "quantile"], 
        labelConfig=labeling.SimpleConfig(relax="relax", tense="tense"),
        scaler = None,
        random_state = None):
    
    file = "/big/f/freiwald/featuresets/local_%d_%.1f.csv" % (10, slidingWindow)
    data = pd.read_csv(file, index_col="id")

    result = []
    for p in participants:
        singledata = dataset.select_participants(data, [p])
    
        train_data = dataset.select_video(singledata, videos_train)
        test_data = dataset.select_video(singledata, videos_test)

        train_data = dataset.select_sensors(train_data, sensors)
        test_data = dataset.select_sensors(test_data, sensors)

        train_data = dataset.select_features(train_data, features)
        test_data = dataset.select_features(test_data, features)

        train_x = shuffle(train_data, random_state=random_state)
        test_x = shuffle(test_data, random_state=random_state)

        train_y = labeling.get_label(train_x, labelConfig)
        test_y = labeling.get_label(test_x, labelConfig)
        
        result.append( (train_x, test_x, train_y, test_y, p) )
        
    return result
