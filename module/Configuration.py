class Configuration:
    type = "orientation"
    raw = "ned"
    window_size = 10
    window_step = 10
    remove_seconds = 0
    use_diff = False
    features = []
    chanels = []
    sensors = []

    def __init__(self,
                 type="orientation",
                 raw="ned",
                 window_size=10,
                 window_step=10,
                 remove_seconds=0,
                 use_diff=False,
                 features=['standard_deviation', 'mean', 'count_above_mean', 'count_below_mean', 'abs_energy', "abs_integral", "velocity", "angular_velocity"],
                 chanels=["x", "y", "z", "w"],  # "q" for combined chanels
                 sensors=["Pelvis", "Chest", "Head", "RightLowerArm", "RightUpperArm", "LeftLowerArm", "LeftUpperArm"]):

        self.type = type
        self.raw = raw
        self.window_size = window_size
        self.window_step = window_step
        self.remove_seconds = remove_seconds
        self.use_diff = use_diff
        self.features = features
        self.chanels = chanels
        self.sensors = sensors

    def __repr__(self):
        return 'Config(\
            \n type=%s,\
            \n raw=%s,\
            \n window_size=%s,\
            \n window_step=%s,\
            \n remove_seconds=%s,\
            \n use_diff=%s,\
            \n features=%s,\
            \n chanels=%s,\
            \n sensors=%s)' \
            % (
                self.type,
                self.raw,
                self.window_size,
                self.window_step,
                self.remove_seconds,
                str(self.use_diff),
                self.features,
                self.chanels,
                self.sensors
                )
