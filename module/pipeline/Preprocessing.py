import os
import pandas as pd
import math

RAW_DIR = "./data/raw"


def process_orientation(dir, file, processes, use_diff):
    participant = file.split("_")[0]
    video = file.split("_")[1]

    df = pd.read_csv(dir + os.sep + file, sep=";")
    df = df.loc[:, df.columns.map(lambda column: "_orientation_q_" in column or column == "time")]

    if use_diff:
        df.loc[:, df.columns.map(lambda column: "_orientation_q_" in column)] = df.loc[:, df.columns.map(lambda column: "_orientation_q_" in column)].diff()
        df = df.fillna(0)

    for process in processes:
        df = process.process(df, participant, video)


    # rename columns
    df.columns = list(map(lambda c: c.split("_")[0]+"_"+c.split("_")[3] if c != "time" else c , df.columns))

    return df


class Preprocessing():
    '''
    Input ist ein Ordner mit Files pro Video/Proband (P01_IT.csv)

    Output sind strukturierter Datensatz
    '''
    def __init__(self, type="orientation", raw="local", use_diff=False, processes=[]):
        self.input = RAW_DIR + os.sep + raw
        self.type = type
        self.processes = processes
        self.use_diff = use_diff

    def execute(self):
        dfs = self.read_input()
        return dfs

    def read_input(self):
        files = [file for file in os.listdir(self.input)]
        dirs = [self.input for i in range(len(files))]
        ps = [self.processes for i in range(len(files))]
        use_diff = [self.use_diff for i in range(len(files))]

        #names = [self.name for i in range(len(files))]


        if self.type == "orientation":
            if self.client is None:
                results = []
                for i in range(len(dirs)):
                    results.append(process_orientation(dirs[i], files[i], ps[i], use_diff[i]))
            else:
                futures = self.client.map(process_orientation, dirs, files, ps, use_diff)
                results = self.client.gather(futures)
        return results

    def config(self):
        removeSensors = list(filter(lambda ps: isinstance(ps, RemoveSensors), self.processes))
        lowPassFilter = list(filter(lambda ps: isinstance(ps, LowPassFilter), self.processes))
        slidingWindow = list(filter(lambda ps: isinstance(ps, SlidingWindow), self.processes))
        removeFirstWindows = list(filter(lambda ps: isinstance(ps, RemoveFirstWindows), self.processes))

        return {
            "Type": self.type,
            "Input": self.input,
            "RemoveSensors": removeSensors[0].config() if len(removeSensors) > 0 else None,
            "LowPassFilter": lowPassFilter[0].config() if len(lowPassFilter) > 0 else None,
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


class LowPassFilter(PreprocessingStep):

    def __init__(self, hrange, hbias):
        self.hrange = hrange
        self.hbias = hbias


    def process(self, df, participant=None, video=None):
        print("Process LowPassFilter")
        df = df.sort_values(by="time")

        ## get all sensors
        sensors = list(set(df.columns.map(lambda row: row if row == "time" else (row.split("_")[0]))))
        sensors.remove("time")

        low  = max(min(self.hbias - (self.hrange/2), 1), 0)
        high = max(min(self.hbias + (self.hrange/2), 1), 0)
        hrangeLimited = high - low
        y = {}
        result = {}
        for sensor in sensors:
            yrow = df.iloc[[0]][["%s_orientation_q_w"%sensor, "%s_orientation_q_x"%sensor, "%s_orientation_q_y"%sensor, "%s_orientation_q_z"%sensor]]
            y[sensor] = Quaternion(yrow["%s_orientation_q_w"%sensor], yrow["%s_orientation_q_x"%sensor], yrow["%s_orientation_q_y"%sensor], yrow["%s_orientation_q_z"%sensor])
            if y[sensor].norm == 0.0:
                y[sensor] = Quaternion(1,0,0,0)
            if sensor not in result:
                result[sensor] = []
            result[sensor].append(list(y[sensor]))

        for i in range(1, len(df)):
            rowresult = []
            for sensor in sensors:
                xrow = df.iloc[[i]][["%s_orientation_q_w"%sensor, "%s_orientation_q_x"%sensor, "%s_orientation_q_y"%sensor, "%s_orientation_q_z"%sensor]]
                x = Quaternion(xrow["%s_orientation_q_w"%sensor], xrow["%s_orientation_q_x"%sensor], xrow["%s_orientation_q_y"%sensor], xrow["%s_orientation_q_z"%sensor])
                if x.norm == 0.0:
                    x = Quaternion(1,0,0,0)
                d = Quaternion.distance(y[sensor], x)
                hlpf = d/math.pi * hrangeLimited + low
                y[sensor] = Quaternion.slerp(y[sensor], x, amount=hlpf)
                result[sensor].append(list(y[sensor]))

        dfs = []
        dfs.append(pd.DataFrame({"time":df.time}))
        for sensor in sensors:
            newdf = pd.DataFrame(result[sensor], columns=["%s_orientation_q_%s"%(sensor,s) for s in list("wxyz")])
            dfs.append(newdf)
        newdf = pd.concat(dfs, axis=1, join='inner')
        return newdf

    def config(self):
        return {"hrange": self.hrange, "hbias": self.hbias}


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
            tmpdf["id"] = "%s_%s_%03d" % (participant, video, int(i / 1000))
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
