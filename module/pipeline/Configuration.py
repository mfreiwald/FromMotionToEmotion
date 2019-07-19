class Configuration:
    window_size = 10
    window_step = 10
    remove_seconds = 10
    features = []
    chanels = []
    sensors = []
    
    def __init__(self,
                 window_size = 10, 
                 window_step = 10, 
                 remove_seconds = 0,
                 features = ['standard_deviation', 'mean', 'count_above_mean', 'count_below_mean', 'abs_energy', "abs_integral", "velocity", "angular_velocity"],
                 chanels = ["x", "y", "z", "w"],  # "q" for combined chanels
                 sensors = ["Pelvis", "Chest", "Head", "RightLowerArm", "RightUpperArm", "LeftLowerArm", "LeftUpperArm"]):
        self.window_size = window_size
        self.window_step = window_step
        self.remove_seconds = remove_seconds
        self.features = features
        self.chanels = chanels
        self.sensors = sensors
        
    def __repr__(self):
        return 'Config(\n window_size=%s,\n window_step=%s,\n remove_seconds=%s,\n features=%s,\n chanels=%s,\n sensors=%s)' % (self.window_size, self.window_step, self.remove_seconds, self.features, self.chanels, self.sensors)