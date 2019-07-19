from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import time, math, uuid, inspect, json
from pyquaternion import Quaternion
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, clear_output
from sklearn.metrics import f1_score
from sklearn import metrics
from IPython.display import HTML, display
import tabulate
from . import validation, labeling, dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
import math
import sys
import seaborn as sns
import traceback


matplotlib.rcParams['figure.figsize'] = (12, 6)

import warnings
warnings.filterwarnings('ignore')
import logging
logging.captureWarnings(True)

class Pipeline:
    
    def __init__(self, preprocess, featureEngineering, featureSelection, crossValidation, classification, random_state=None, id=0):
        
        self.random_state = random_state
        
        self.start_cluster()
        self.wait_for_cluster(id)
        
        try:
        
            # check for existing parameters, else execute
            preConfig = preprocess.config()

            preName = self.checkConfig("preprocess", preConfig)
            if not preName:
                clear_output(wait=True)
                display("(%d) Preprocessing..." % id)
        
                preName = str(uuid.uuid4())
                preprocess.client = Client(self.get_scheduler().address)
                preprocess.name = preName
                preprocess.execute()
                configFile = "./pipelineConfig/preprocess/%s.json" % preName
                with open(configFile, 'w') as outfile:  
                    json.dump(preConfig, outfile)



            feConfig = {"preprocess": preConfig, "featureEngineering": featureEngineering.config()}

            feName = self.checkConfig("features", feConfig)
            if not feName:
                clear_output(wait=True)
                display("(%d) Feature Engineering..." % id)
                
                feName = str(uuid.uuid4())
                featureEngineering.client = Client(self.get_scheduler().address)
                featureEngineering.name = feName
                featureEngineering.preprocessName = preName
                featureEngineering.execute()
                configFile = "./pipelineConfig/features/%s.json" % feName
                with open(configFile, 'w') as outfile:  
                    json.dump(feConfig, outfile)


            clear_output(wait=True)
            display("(%d) Load Features..." % id)        
            df = pd.read_csv("/big/f/freiwald/features/%s.csv" % feName, index_col="id")


            '''
            fsConfig = {"preprocess": preConfig, "featureEngineering": feConfig, "featureSelection": featureSelection.config()}
            fsName = self.checkConfig("featureSelection", feConfig)


            if not fsName:
                fsName = str(uuid.uuid4())
                featureSelection.client = Client(self.get_scheduler().address)
                featureSelection.name = fsName
                featureSelection.featuresName = feName
                featureSelection.execute()
                configFile = "./pipelineConfig/featureSelection/%s.json" % fsName
                with open(configFile, 'w') as outfile:  
                    json.dump(fsConfig, outfile)
            '''
            
            featureSelection.client
            featureSelection.df = df
            featureSelection.random_state = self.random_state
            df = featureSelection.execute()
            
            clear_output(wait=True)
            display("(%d) Create Cross Validation..." % id )
            cv = ParticipantCrossValidation(df,
                                           crossValidation.participants,
                                           crossValidation.labelConfig,
                                           crossValidation.shuffle_videos,
                                           crossValidation.videos_train,
                                           crossValidation.videos_test,
                                           self.random_state)
            
            cv.client = Client(self.get_scheduler().address)
            splits = cv.split()

            clear_output(wait=True)
            display("(%d) Test Classifiers..." % id)
            
            classification.client = Client(self.get_scheduler().address)
            classification.random_state = self.random_state
            classification.splits = splits
        
            self.results, test_y, predict_y = classification.execute()
        
            self.report = {}
            for key in test_y.keys():
                self.report[key] = metrics.classification_report(test_y[key], predict_y[key])
            
        
            self.scores = {}
            
            for key in self.results.keys():
                self.scores[key] = {}
                self.scores[key]["all"] = np.array(list(map(lambda p: self.results[key][p]["score"], self.results[key])))
                self.scores[key]["f1"] = np.array(list(map(lambda p: self.results[key][p]["f1"], self.results[key])))
                #self.scores[key]["World"] = np.array(list(map(lambda p: self.results[key][p]["videos"]["World"], self.results[key])))
                #self.scores[key]["Relax"] = np.array(list(map(lambda p: self.results[key][p]["videos"]["Relax"], self.results[key])))
                #self.scores[key]["IT"] = np.array(list(map(lambda p: self.results[key][p]["videos"]["IT"], self.results[key])))
                #self.scores[key]["Nun"] = np.array(list(map(lambda p: self.results[key][p]["videos"]["Nun"], self.results[key])))
            
            
            clear_output(wait=True)
        except:
            print("Unexpected error:", sys.exc_info())
            traceback.print_exc()
        
        finally:
            self.cluster.close()
            self.preName = preName
            self.feName = feName
        
    
    def start_cluster(self):
        self.cluster = SLURMCluster(
            memory='16g',
            processes=1,
            cores=1,
            death_timeout=200,
            walltime="168:00:00",
            job_extra=['--partition=Gobi'])
        self.cluster.start_workers(25)
    
    def wait_for_cluster(self, id):
        while len(self.cluster.running_jobs) < 4:
            clear_output(wait=True)
            display("(%d) Wait for cluster (%d/25)" % (id, len(self.cluster.running_jobs)))
            time.sleep(2)
        clear_output(wait=True)
        display("(%d) Wait for cluster (%d/25)" % (id, len(self.cluster.running_jobs)))
        
        
        
    def get_scheduler(self):
        return self.cluster.scheduler
    
    
    def checkConfig(self, step, config):
        for configFile in os.listdir("./pipelineConfig/%s"%step):
            with open("./pipelineConfig/%s/%s"%(step,configFile)) as json_file:  
                otherConfig = json.load(json_file)
                if otherConfig == config:
                    return configFile[:-5]
        return None
    
    
    def load_featureset(self):
        return pd.read_csv("/big/f/freiwald/features/%s.csv" % self.feName, index_col="id")
    
    def stats(self):
        table = [
            np.append("", list(self.scores.keys())),
            np.append("Acc:", list(map(lambda key: "%0.2f (+/- %0.2f) " % (self.scores[key]["all"].mean(), self.scores[key]["all"].std() * 2), self.scores.keys()))),
            np.append("F1:", list(map(lambda key: "%0.2f (+/- %0.2f) " % (self.scores[key]["f1"].mean(), self.scores[key]["f1"].std() * 2), self.scores.keys()))),
            ]
        display(HTML(tabulate.tabulate(table, tablefmt='html')))
        
    def accuracy(self):
        result = {}
        for key in self.scores.keys():
            result[key] = self.scores[key]["all"].mean()
        return result
    
    def std(self):
        result = {}
        for key in self.scores.keys():
            result[key] = self.scores[key]["all"].std()
        return result
    
    def f1_score(self):
        result = {}
        for key in self.scores.keys():
            result[key] = self.scores[key]["f1"].mean()
        return result

    
    
class PipelineObject:
    client = None
    name = None
    
    def execute(self):
        pass
    
    
def process(dir, file, processes, name):
    participant = file.split("_")[0]
    video = file.split("_")[1]
    
    df = pd.read_csv(dir + os.sep + file, sep=";")
    df = df.loc[:, df.columns.map(lambda column: "_position_v_" in column or column == "time")]
    
    df.loc[:, df.columns.map(lambda column: "_position_v_" in column)] = df.loc[:, df.columns.map(lambda column: "_position_v_" in column)].diff()
    df = df.fillna(0)

    for process in processes:
        df = process.process(df, participant, video)
    
    
    # rename columns
    df.columns = list(map(lambda c: c.split("_")[0]+"_"+c.split("_")[3] if c != "time" else c , df.columns))

    destDir = "/big/f/freiwald/preprocessing/%s" % name
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    destFile = "%s/%s_%s.csv" % (destDir, participant, video)
    df.to_csv(destFile)
    return df


class Preprocessing(PipelineObject):
    '''
    Input ist ein Ordner mit Files pro Video/Proband (P01_IT.csv)
    
    Output sind strukturierter Datensatz
    '''
    def __init__(self, raw="positions", processes=[]):
        self.input = "/big/f/freiwald/raw" + os.sep + raw
        self.processes = processes
        
    def execute(self):
        dfs = self.read_input()
        #for df in dfs:
            #print(df.index)
            # df.to_csv("/big/f/freiwald/proprocessing/")
    
    def read_input(self):
        files = [file for file in os.listdir(self.input)]
        dirs = [self.input for i in range(len(files))]
        ps = [self.processes for i in range(len(files))]
        names = [self.name for i in range(len(files))]
        futures = self.client.map(process, dirs, files, ps, names)
        results = self.client.gather(futures)
        #results = [process(dirs[0], files[0], ps[0], names[0])]
        return results
    
    def config(self):
        removeSensors = list(filter(lambda ps: isinstance(ps, RemoveSensors), self.processes))
        slidingWindow = list(filter(lambda ps: isinstance(ps, SlidingWindow), self.processes))
        removeFirstWindows = list(filter(lambda ps: isinstance(ps, RemoveFirstWindows), self.processes))

        return {
            "Type": "Diff3DPosition42",
            "Input": self.input,
            "RemoveSensors": removeSensors[0].config() if len(removeSensors) > 0 else None,
            "SlidingWindow": slidingWindow[0].config() if len(slidingWindow) > 0 else None,
            "RemoveFirstWindows": removeFirstWindows[0].config() if len(removeFirstWindows) > 0 else None
        }
        
class PreprocessingStep:
    
    def process(self, df, participant=None, video=None):
        return df
    
    def config(self):
        return None
    
    
class RemoveSensors(PreprocessingStep):
    
    def __init__(self, only=None):
        self.sensors = only
        
    def process(self, df, participant=None, video=None):
        if self.sensors is None: # or set(self.sensors) == set(["Head", "Chest", "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm", "Pelvis"]):
            return df
        else:
            return df.loc[:, df.columns.map(lambda column: column == "time" or any(sensor+"_" in column for sensor in self.sensors))]
        
    def config(self):
        return {"sensors": self.sensors}
    
    
class SlidingWindow(PreprocessingStep):
    
    def __init__(self, size, step):
        self.size = size
        self.step = step
        
        
    def process(self, df, participant=None, video=None):
        print("Process SlidingWindow")
        
        maxTime = df.time.max()
        newdf = None
        for i in range(0, math.ceil(maxTime) - self.size * 1000, int(self.step * 1000)):
            j = i + self.size * 1000
            tmpdf: pd.DataFrame = df.loc[((df.time >= i) & (df.time < j))]
            tmpdf["id"] = "%s_%s_%d" % (participant, video, int(i / 1000))
            if newdf is None:
                newdf = tmpdf
            else:
                newdf = newdf.append(tmpdf)
        return newdf.set_index("id")
    
    def config(self):
        return {"size": self.size, "step": self.step}
    
    
class RemoveFirstWindows(PreprocessingStep):
    
    def __init__(self, seconds):
        self.seconds = seconds
    
    def process(self, df, participant=None, video=None):
        print("Process RemoveFirstWindows")   
        return df.groupby(level=0).filter(lambda g: g.time.min() > (self.seconds*1000))
    
    def config(self):
        return {"seconds": self.seconds}
    
    
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.distribution import MapDistributor

def extract(dir, file, features, name):
    participant = file.split("_")[0]
    video = file.split("_")[1]
    df = pd.read_csv(dir + os.sep + file)
    extracted_features = extract_features(df, column_id="id", column_sort="time", default_fc_parameters = features, distributor=MapDistributor())
    impute(extracted_features)    
    return extracted_features
    
class FeatureEngineering(PipelineObject):
    
    preprocessName = None
    
    def __init__(self, features):
        self.features = features
        
    def execute(self):
        dfs = self.read_input()
        df = pd.concat(dfs)
        df.to_csv("/big/f/freiwald/features/%s.csv"%self.name)
    
    def read_input(self):
        indir = "/big/f/freiwald/preprocessing/%s"%self.preprocessName
        files = os.listdir(indir)
        dirs = [indir for i in range(len(files))]
        fs = [self.features for i in range(len(files))]
        names = [self.name for i in range(len(files))]
        futures = self.client.map(extract, dirs, files, fs, names)
        results = self.client.gather(futures)
        #results = [extract(dirs[0], files[0], fs[0], names[0])]
        return results
    
    def config(self):
        return {"features": self.features}
    
    
from sklearn.decomposition import PCA


class FeatureSelection(PipelineObject):
    
    df = None
    random_state = None
    
    def __init__(self, method=None, args=None):
        self.method = method
        self.args = args
    
    def execute(self):
        if not self.method:
            return self.df
        
        if self.method == "pca" and len(self.args) > 0:
            n = self.args[0]
            # n = min(len(self.df.columns), len(self.df))
            pca = PCA(n_components = n, random_state=self.random_state)
            pca.fit(self.df)
            columns = ['pca_%i' % i for i in range(n)]
            df_pca = pd.DataFrame(pca.transform(self.df), columns=columns, index=self.df.index)
            return df_pca
        
        return self.df
            
        
    def config(self):
        return {"selection": self.method}
        
        
        
from sklearn.model_selection import LeaveOneOut
from module import dataset, labeling
from contextlib import suppress
from random import randint

class CrossValidation(PipelineObject):

    random_state = None
    
    def __init__(self, 
                 participants=np.array(["P%02.d"%d for d in range(1,21) if d != 11]),
                 labelConfig=labeling.SimpleConfig(),
                 shuffle_videos=False,
                 videos_train=[1,2,3,4],
                 videos_test=[1,2,3,4]):
        
        self.participants = participants
        self.labelConfig = labelConfig
        self.shuffle_videos = shuffle_videos
        self.videos_train = videos_train
        self.videos_test = videos_test
        
    def execute(self):
        pass
    
    def config(self):
        return None
    
    
def gen_split(df, participants, videos_train, videos_test, labelConfig, shuffle_videos, random_state, train_index, test_index):
    train_p = participants[train_index]
    test_p = participants[test_index]

    train_data = dataset.select_participants(df, train_p)
    test_data = dataset.select_participants(df, test_p)

    if shuffle_videos:
        # random for every cv step
        train_data = dataset.select_video(train_data, videos_train)
        test_data = dataset.select_video(test_data, videos_test)
    
    
    '''    
    setAllVideos = set([1,2,3,4])
    if set(videos_train) == setAllVideos and set(videos_test) == setAllVideos:
        pass
    else:
        train_data = dataset.select_video(train_data, videos_train)
        test_data = dataset.select_video(test_data, videos_test)
     '''   
        
    train_x = shuffle(train_data, random_state=random_state)
    test_x = shuffle(test_data, random_state=random_state)

    train_y = labeling.get_label(train_x, labelConfig)
    test_y = labeling.get_label(test_x, labelConfig)
        
    return (train_x, test_x, train_y, test_y, test_p[0])

class ParticipantCrossValidation():
    
    client = None
    
    def __init__(self, df, 
                 participants=np.array(["P%02.d"%d for d in range(1,21) if d != 11]),
                 labelConfig=labeling.SimpleConfig(),
                 shuffle_videos=False,
                 videos_train=[1,2,3,4],
                 videos_test=[1,2,3,4],
                 random_state=None):
        
        self.df = df
        self.participants = participants
        self.labelConfig = labelConfig
        self.shuffle_videos = shuffle_videos
        self.videos_train = videos_train
        self.videos_test = videos_test
        self.random_state = random_state
        
    def split(self):
        splits = []
        futures = []
        loo = LeaveOneOut()
        
        np.random.seed(self.random_state)

        videoCombis = [
            [[1,3], [2,4]],
            [[2,4], [1,3]],
            [[1,4], [2,3]],
            [[2,3], [1,4]],
        ]
        
        for train_index, test_index in loo.split(self.participants):
            
            videoComb = videoCombis[np.random.randint(0, len(videoCombis))]
            random_train = videoComb[0]
            random_test = videoComb[1]

            # future = client.submit(func, big_data)    # bad                
            future = self.client.submit(gen_split, self.df, self.participants, random_train, random_test, self.labelConfig, self.shuffle_videos, self.random_state, train_index, test_index)

            
            #big_future = client.scatter(big_data)     # good
            #future = client.submit(func, big_future)  # good
            #big_future = self.client.scatter(self.df, self.participants, self.videos_train, self.videos_test, self.labelConfig, self.random_state, train_index, test_index)
            #future = self.client.submit(gen_split, big_future)

            futures.append(future)
            
        for future in futures:
            splits.append(future.result())
        return splits
    
    
from sklearn import metrics

def train(name, participant, clf, train_x, train_y, test_x, test_y):
    model = clf.fit(train_x, train_y)
    
    
    try:
        importances = dict(zip(train_x.columns, clf.feature_importances_)) 
        
    except AttributeError:
        importances = None
    
    score = model.score(test_x, test_y)

    predict_y = model.predict(test_x)
    
    #predicted_proba = model.predict_proba(test_x)
    
    # precision, recall, thresholds = metrics.precision_recall_curve(test_y, predicted_proba[:, 1])
    
    '''
    threshold_accuracy = {}
    for t in range(0, 50):
        threshold = t / 50.0
        t_predicted = (predicted_proba [:,1] >= threshold).astype('int')
        a = metrics.accuracy_score(test_y, t_predicted)
        p = metrics.precision_score(test_y, t_predicted)
        r = metrics.recall_score(test_y, t_predicted)
        threshold_accuracy[threshold] = 
    '''
    
    conMatrix = metrics.confusion_matrix(test_y, predict_y)
    f1Score = metrics.f1_score(test_y, predict_y, pos_label="relax")
    report = metrics.classification_report(test_y, predict_y)
    
    
    '''
    w_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "World")]
    r_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "Relax")]
    i_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "IT")]
    n_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "Nun")]
        
    w_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "World")]
    r_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "Relax")]
    i_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "IT")]
    n_y = test_y.loc[test_y.index.map(lambda idx: idx.split("_")[1] == "Nun")]
    
    perVideo = [("World", w_x, w_y), ("Relax", r_x, r_y), ("IT", i_x, i_y), ("Nun", n_x, n_y)]
    
    videoScores = {}
    videof1Score = {}
    for video, x, y in perVideo:
        if len(x) > 0:
            videoScores[video] = model.score(x, y)
            p_y = model.predict(x)
            videof1Score[video] = metrics.f1_score(y, p_y, pos_label="relax")
     '''   
    videoScores = {}
    videof1Score = {}
    return name, participant, score, conMatrix, f1Score, videoScores, videof1Score, importances, report, model.classes_, test_y, predict_y


class Classification(PipelineObject):
    
    random_state = None
    splits = None
    
    def __init__(self, clfs, threshold):
        self.clfs = clfs
        self.threshold = threshold
        
        
    def execute(self):
        
        self.results = {name:{} for name in self.clfs.keys()}
        
        futures = []
        for (train_x, test_x, train_y, test_y, participant) in self.splits:
            for name, clf in self.clfs.items():
                
                try:
                    clf.random_state = self.random_state
                except AttributeError:
                    pass
                
                if participant not in self.results[name]:
                    self.results[name][participant] = {}
            
                self.results[name][participant]["train_size"] = len(train_x)
                self.results[name][participant]["test_size"] = len(test_x)
                '''                
                w_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "World")]
                r_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "Relax")]
                i_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "IT")]
                n_x = test_x.loc[test_x.index.map(lambda idx: idx.split("_")[1] == "Nun")]
    
                self.results[name][participant]["test_world_size"] = len(w_x)
                self.results[name][participant]["test_relax_size"] = len(r_x)
                self.results[name][participant]["test_it_size"] = len(i_x)
                self.results[name][participant]["test_nun_size"] = len(n_x)
                '''
                future = self.client.submit(train, name, participant, clf, train_x, train_y, test_x, test_y)
                futures.append(future)
                
        
        all_test_y = {}
        all_predict_y = {}
        
        for future in futures:
            name, participant, score, conMatrix, f1Score, videoScores, videof1Score, importances, report, classes_, test_y, predict_y = future.result()
            if name not in all_test_y:
                all_test_y[name] = []
            if name not in all_predict_y:
                all_predict_y[name] = []
                
            self.results[name][participant]["score"] = score
            self.results[name][participant]["matrix"] = conMatrix
            self.results[name][participant]["f1"] = f1Score
            self.results[name][participant]["videos"] = videoScores
            self.results[name][participant]["videosf1"] = videof1Score
            self.results[name][participant]["importances"] = importances
            self.results[name][participant]["report"] = report
            self.results[name][participant]["classes_"] = classes_
            all_test_y[name].extend(test_y)
            all_predict_y[name].extend(predict_y)


        return self.results, all_test_y, all_predict_y
    
    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.feature_extraction.settings import EfficientFCParameters


all_features = {
        "median": None,
        'mean': None,
        'standard_deviation': None,
        'maximum': None,
        'minimum': None,
        'count_above_mean': None,
        'count_below_mean': None,
        'longest_strike_below_mean': None,
        'longest_strike_above_mean': None,
        'abs_energy': None,
        'cid_ce': [{'normalize': True}, {'normalize': False}],
        'number_peaks': [
            {'n': 1}, 
            {'n': 3}, 
            {'n': 5}, 
            {'n': 10}, 
            #{'n': 50}
        ],
        'autocorrelation': [
            #{'lag': 0},
            {'lag': 1},
            {'lag': 2},
            {'lag': 3},
            {'lag': 4},
            {'lag': 5},
            {'lag': 6},
            {'lag': 7},
            {'lag': 8},
            {'lag': 9}
        ],
        'quantile': [
            {'q': 0.1},
            {'q': 0.2},
            {'q': 0.3},
            {'q': 0.4},
            {'q': 0.6},
            {'q': 0.7},
            {'q': 0.8},
            {'q': 0.9}
        ],  
        'spkt_welch_density': [
            {'coeff': 2}, 
            {'coeff': 5}, 
            {'coeff': 8}
        ],
        'agg_autocorrelation':[ 
            {'f_agg': 'mean'}, 
            {'f_agg': 'median'}, 
            {'f_agg': 'var'}
        ]
    }


min_features = {
            'standard_deviation': None, 
            'count_above_mean': None, 
            'count_below_mean': None,
            'mean': None, 
            'maximum': None, 
            'minimum': None
        }


all_sensors = ["Head", "Chest", "RightUpperArm", "RightLowerArm", "Pelvis"]

def evaluate(features, sensors, lpf=None, window_size=10, window_step=5.0, remove_seconds=30, shuffle_videos=True, labelConfig=labeling.SimpleConfig()):
    pipes = []
    for i in range(10):
        pipe = Pipeline(Preprocessing(processes=[
                        RemoveSensors(only=sensors),
                        SlidingWindow(size=window_size, step=window_step),
                        RemoveFirstWindows(seconds=remove_seconds),
                       ]),
                       FeatureEngineering(features), # {'standard_deviation': None, 'count_above_mean': None, 'count_below_mean': None,}),
                       FeatureSelection(None), # "pca", [20]),
                       CrossValidation(
                           labelConfig=labelConfig, # RankingThresholdConfig(threshold=2.5)
                           shuffle_videos=shuffle_videos
                       ),
                       Classification({
                           "RandomForest": RandomForestClassifier(
                                n_estimators=1800, 
                                max_features="auto", 
                                max_depth=40, 
                                min_samples_split=5,
                                min_samples_leaf=1,
                                bootstrap=True,
                                n_jobs=-1),
                           "AdaBoost": AdaBoostClassifier(
                               n_estimators=100),
                           "SVC": SVC(),
                           "Most Frequent": DummyClassifier(strategy="most_frequent"),
                           "Random": DummyClassifier(strategy="uniform"),
                       },
                       threshold=None),
                       random_state=None,
                       id=i
                     )
        pipes.append(pipe)
        time.sleep(10)

    table = []
    table.append(["", "Accuracy", "F1-Score"])
    p0 = pipes[0]
    keys = p0.scores.keys()
    for key in keys:
        accPerCLF = np.mean(list(map(lambda p: p.accuracy()[key], pipes)))    
        stdPerCLF = np.mean(list(map(lambda p: p.std()[key], pipes)))    
        f1sPerCLF = np.mean(list(map(lambda p: p.f1_score()[key], pipes)))
        table.append([key, "%0.2f +- %0.2f" % (accPerCLF, stdPerCLF), "%0.2f" % f1sPerCLF])
    display(HTML(tabulate.tabulate(table, tablefmt='html')))

    row = []
    for idx, p in enumerate(pipes):
        for clf in p.results.keys():
            for part in p.results[clf].keys():
                score = p.results[clf][part]["score"]
                row.append([idx, clf, part, score])
    df = pd.DataFrame(row, columns=["pipeline", "clf", "participant", "accuracy"])
    plt.ylim(0, 1)
    sns.boxplot(x='clf', y='accuracy', data=df.groupby(by=["clf", "participant"]).accuracy.mean().reset_index())
    
    return pipes


