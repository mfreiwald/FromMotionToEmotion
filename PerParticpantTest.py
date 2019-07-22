from module.Configuration import Configuration
from module.Evaluation import Evaluation
# from module.Cluster import Cluster
from dask.distributed import Client
import pandas as pd
import numpy as np
import logging
from module import labeling
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from collections import defaultdict
from sklearn.utils import shuffle


file = "./data/metadata/questionnaire.csv"
metadf = pd.read_csv(file, sep=";", index_col="name")


def _video_of(name, nrs):
    return list(map(lambda nr: metadf.loc[name]["video%d"%nr], nrs))


def main():
    conf = Configuration(window_size=20, window_step=5.0)
    print(conf)

    c = Client()

    # On a SLURM Network, you can call:
    # clu = Cluster()
    # c = clu.cli
    # Check module/Cluster.py for more details

    eva = Evaluation(c, conf)

    logging.getLogger('distributed.utils_perf').setLevel(logging.CRITICAL)

    df1 = eva.make_preprocessing()
    df2 = eva.make_feature_engineering(df1)
    df3 = eva.make_selection(df2)

    participants = np.array(sorted(set(map(lambda i: i.split("_")[0], df3.index))))

    results = defaultdict(list)

    for labelConfig in [
        labeling.SimpleConfig() #, labeling.RankingThresholdConfig(), labeling.OnlineConfig(), labeling.ExpertConfig()
    ]:
        labelConfig.__class__.__name__

        for videos in [
            [[1,3], [2,4]],
            [[2,4], [1,3]],
            [[1,4], [2,3]],
            [[2,3], [1,4]]
        ]:
            videos_train = videos[0]
            videos_test = videos[1]

            results["Config"].append(labelConfig.__class__.__name__)
            results["Train"].append(str(videos_train))

            # results["Train"].append(videos_train)
            for part in participants:
                x = df3.loc[df3.index.map(lambda x: any(arg+"_" in x for arg in part))]

                mean_score = []
                for state in range(10,110,10):

                    train_x = x.loc[x.index.map(lambda idx: any(video in idx for video in _video_of(idx.split("_")[0], videos_train) ))]
                    test_x = x.loc[x.index.map(lambda idx: any(video in idx for video in _video_of(idx.split("_")[0], videos_test) ))]

                    train_x = shuffle(train_x, random_state=state)
                    test_x = shuffle(test_x, random_state=state)

                    train_y = labeling.get_label(train_x, labelConfig)
                    test_y = labeling.get_label(test_x, labelConfig)

                    clf = RandomForestClassifier(
                        n_estimators=100,
                        n_jobs=-1)
                    clf.state = state
                    model = clf.fit(train_x, train_y)
                    yp_test = model.predict(test_x)
                    score = metrics.accuracy_score(test_y, yp_test)
                    mean_score.append(score)

                results[part].append(np.mean(mean_score))

    dd = pd.DataFrame(results)
    dd
    # clu.close()


if __name__ == "__main__":
    main()
