from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
from sklearn import metrics
from module import labeling

file = "./data/metadata/questionnaire.csv"
metadf = pd.read_csv(file, sep=";", index_col="name")


def _video_of(name, nrs):
    return list(map(lambda nr: metadf.loc[name]["video%d"%nr], nrs))


classifiers = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100),
    "SVC": SVC(),
    "MostFrequent": DummyClassifier(strategy="most_frequent"),
    "Random": DummyClassifier(strategy="uniform"),
}

labelConfig = labeling.SimpleConfig()             #labeling.RankingThresholdConfig() #  RankingThresholdConfig()  SimpleConfig()  OnlineConfig()  ExpertConfig()
testlabelConfig = labeling.SimpleConfig()         #labeling.RankingThresholdConfig() #  RankingThresholdConfig()  SimpleConfig()  OnlineConfig()

videoCombis = [
    [[1,3], [2,4]],
    [[2,4], [1,3]],
    [[1,4], [2,3]],
    [[2,3], [1,4]],

    #[[2,4],[2,4]],
    #[[1,2,3,4],[1,2,3,4]]
]

def run_clf(clf, train_x, train_y, test_x, test_y, state):
    clf.random_state = state
    model = clf.fit(train_x, train_y)
    test_yp = model.predict(test_x)

    score = metrics.accuracy_score(test_y, test_yp)
    cm = metrics.confusion_matrix(test_y, test_yp, labels=["tense", "relax"]) # ACHTUNG: Geht nur bei diesen zwei Label !!!

    verteilung_klassen_true = pd.Series(test_y).value_counts(normalize=True)
    verteilung_klassen_pred = pd.Series(test_yp).value_counts(normalize=True)
    tn_tense_true, fp_tense_false, fn_relax_false, tp_relax_true = cm.ravel() # tn, fp, fn, tp

    info = {}
    info["verteilung_klassen_true_relax"] = verteilung_klassen_true["relax"] if "relax" in verteilung_klassen_true else 0
    info["verteilung_klassen_true_tense"] = verteilung_klassen_true["tense"] if "tense" in verteilung_klassen_true else 0

    info["verteilung_klassen_pred_relax"] = verteilung_klassen_pred["relax"] if "relax" in verteilung_klassen_pred else 0
    info["verteilung_klassen_pred_tense"] = verteilung_klassen_pred["tense"] if "tense" in verteilung_klassen_pred else 0

    info["tn_tense_true"] = tn_tense_true
    info["fp_tense_false"] = fp_tense_false
    info["fn_relax_false"] = fn_relax_false
    info["tp_relax_true"] = tp_relax_true

    importances = {}
    probas_df = None
    if type(clf) is RandomForestClassifier:
        importances = dict(zip(train_x.columns, model.feature_importances_))

        test_yp_proba = model.predict_proba(test_x)
        proba_classes = model.classes_
        # test_x.index = Liste der Video-Abschnitte
        # test_yp_proba = Liste mit Tuple, Wahrscheinlich der Klassen
        # proba_classes = Reihenfolge der Klassen im proba Tuple
        l = []
        for proba in test_yp_proba:
            l.append(dict(zip(proba_classes, proba)))
        probas_df = pd.DataFrame(dict( zip(list(test_x.index), l))).transpose()


    return score, cm, info, importances, probas_df

def run_class(data, splits):
    cv_scores = {}

    participants = np.array(sorted(set(map(lambda i: i.split("_")[0], data.index))))

    train_index = splits[0]
    test_index = splits[1]

    per_participant_train_size_relax = []
    per_participant_train_size_tense = []
    per_participant_test_size_relax = []
    per_participant_test_size_tense = []

    info_df = None
    importances_df = None
    proba_dfs = []


    videos_score = defaultdict(list)

    for videos in videoCombis:
        videos_train = videos[0]
        videos_test = videos[1]

        train_p = participants[train_index]
        test_p = participants[test_index]

        train_x = data.loc[data.index.map(lambda x: any(arg+"_" in x for arg in train_p))]
        test_x = data.loc[data.index.map(lambda x: any(arg+"_" in x for arg in test_p))]

        train_x = train_x.loc[train_x.index.map(lambda idx: any(video in idx for video in _video_of(idx.split("_")[0], videos_train) ))]
        test_x = test_x.loc[test_x.index.map(lambda idx: any(video in idx for video in _video_of(idx.split("_")[0], videos_test) ))]

        train_size = None
        test_size = None
        mean_score = defaultdict(list)
        for state in range(10,110,10):
            train_x_s = shuffle(train_x, random_state=state)
            test_x_s = shuffle(test_x, random_state=state)

            train_y = labeling.get_label(train_x_s, labelConfig)
            test_y = labeling.get_label(test_x_s, testlabelConfig)

            train_size = train_y.value_counts(normalize=False)
            test_size = test_y.value_counts(normalize=False)

            for clf_name, clf in classifiers.items():
                score, cm, info, importances, probas_df = run_clf(clf, train_x_s, train_y, test_x_s, test_y, state)
                mean_score[clf_name].append(score)


                if clf_name is "RandomForest":

                    if info_df is None:
                        info_df = pd.DataFrame(info, index=['42',])
                        importances_df = pd.DataFrame([importances.values()], columns=importances.keys())
                    else:
                        info_df = info_df.append(info, ignore_index=True)
                        importances_df = importances_df.append(pd.DataFrame([importances.values()], columns=importances.keys()), ignore_index=True)

                    proba_dfs.append(probas_df)


        per_participant_train_size_relax.append(train_size["relax"] if "relax" in train_size else 0)
        per_participant_train_size_tense.append(train_size["tense"] if "tense" in train_size else 0)
        per_participant_test_size_relax.append(test_size["relax"] if "relax" in test_size else 0)
        per_participant_test_size_tense.append(test_size["tense"] if "tense" in test_size else 0)

        for clf, scores in mean_score.items():
            videos_score[clf].append(np.mean(scores))
        #test_p[0], videos_test, np.mean(mean_score)

    for clf, scores in videos_score.items():
        cv_scores[clf] = np.mean(scores)
        #print(test_p[0], clf, np.mean(scores))

    result = {}
    result["participant"] = participants[test_index]
    result["scores"] = cv_scores
    result["train_size_relax"] = np.mean(per_participant_train_size_relax)
    result["train_size_tense"] = np.mean(per_participant_train_size_tense)
    result["test_size_realx"] = np.mean(per_participant_test_size_relax)
    result["test_size_tense"] = np.mean(per_participant_test_size_tense)
    result["info"] = info_df.mean()
    result["importances"] = importances_df.mean()
    result["probas"] = pd.concat(proba_dfs).groupby(level=0).mean()


    return result



class Classification():

    def __init__(self):
        pass

    def execute(self, data):
        print("Make Classification with ", labelConfig)
        participants = np.array(sorted(set(map(lambda i: i.split("_")[0], data.index))))
        splits = list(LeaveOneOut().split(participants))


        datas = [data for i in range(len(splits))]

        big_future_data = self.client.scatter(datas)

        futures = self.client.map(run_class, big_future_data, splits)
        all_cv_results = self.client.gather(futures)

        cv_scores = defaultdict(list)
        train_size_relax = []
        train_size_tense = []
        test_size_realx = []
        test_size_tense = []

        for dict_clf_score in all_cv_results:
            for name, score in dict_clf_score["scores"].items():
                cv_scores[name].append(score)
            train_size_relax.append(dict_clf_score["train_size_relax"])
            train_size_tense.append(dict_clf_score["train_size_tense"])
            test_size_realx.append(dict_clf_score["test_size_realx"])
            test_size_tense.append(dict_clf_score["test_size_tense"])

        #for split in splits:
        #    dict_clf_score = run_class(split, data)
        #    for name, score in dict_clf_score.items():
        #        cv_scores[name].append(score)

        results = {}
        for name, scores in cv_scores.items():
            print(name, np.mean(scores), np.std(scores))
            results[name] = [np.mean(scores), np.std(scores)]

        sizes = {}
        sizes["train_size_relax"] = np.mean(train_size_relax)
        sizes["train_size_tense"] = np.mean(train_size_tense)
        sizes["test_size_realx"] = np.mean(test_size_realx)
        sizes["test_size_tense"] = np.mean(test_size_tense)

        #### FEHLER::::
        ###  ICH HAB DIE UNTERSCHIEDLICHEN CLASSIFIER NICHT BEACHTET bei der INFO!!!
        ### Am besten einfach nur den RandomForest verwenden
        info_dataframe = None
        for result in all_cv_results:
            result["info"]["score"] = result["scores"]["RandomForest"]
            if info_dataframe is None:
                info_dataframe = pd.DataFrame(result["info"], columns=[result["participant"]]).transpose()
            else:
                info_dataframe = info_dataframe.append(pd.DataFrame(result["info"], columns=[result["participant"]]).transpose())

        importances_dataframe = None
        for result in all_cv_results:
            if importances_dataframe is None:
                importances_dataframe = pd.DataFrame(result["importances"], columns=[result["participant"]]).transpose()
            else:
                importances_dataframe = importances_dataframe.append(pd.DataFrame(result["importances"], columns=[result["participant"]]).transpose())

        all_cv_probas = {}
        for result in all_cv_results:
            all_cv_probas[result["participant"][0]] = result["probas"] # dataframe


        return results, sizes, info_dataframe, importances_dataframe, all_cv_probas
