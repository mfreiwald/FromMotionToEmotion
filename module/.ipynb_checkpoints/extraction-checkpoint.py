from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from module.slurmDaskDistributor import SlurmDaskDistributor
from module import cluster, ifttt
from datetime import datetime
import pandas as pd
import sys
import os
import os.path

def extract_tsfresh(sourcePath, destinationPath):
    if not os.path.exists(destinationPath):
        os.makedirs(destinationPath)

    c = cluster.start_cluster(workers=300)

    files = os.listdir(sourcePath)
    nrFiles = len(files)
    allStart = datetime.now()
    for idx, file in enumerate(files):
        idx += 1

        if os.path.isfile(destinationPath + os.sep + file):
            continue

        df = pd.read_csv(sourcePath + os.sep + file, sep=";")
        start = datetime.now()
        print(file, "%d/%d" % (idx, nrFiles))
        print(file, start.strftime("%a, %d %B %Y %I:%M:%S"))

        Distributor = SlurmDaskDistributor(c)
        extracted_features = extract_features(df, column_id="id", column_sort="time", distributor=Distributor)
        impute(extracted_features)
        end = datetime.now()
        delta = end-start
        print(file, end.strftime("%a, %d %B %Y %I:%M:%S"))
        print("Time", str(delta))
        print("")
        extracted_features.to_csv(destinationPath + os.sep + file, sep=";")
        ifttt.send("ExtFeat %s" % file, "%d/%d" % (idx, nrFiles))

    allEnd = datetime.now()
    allDelta = allEnd-allStart

    print("Start:", allStart.strftime("%a, %d %B %Y %I:%M:%S"))
    print("End:", allEnd.strftime("%a, %d %B %Y %I:%M:%S"))
    print("Time:", str(allDelta))
    ifttt.send("ExtFeat", str(allDelta))


def extract_velocity():
    pass


def extract_angular_velocity():
    pass


