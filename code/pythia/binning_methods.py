# list of methods for bin splitting


class DynamicProgramming:
    def __init__(self,
                 max_nof_bins: int = 20,
                 min_points_per_bin: int = 10,
                 discount: float = 0.3):
        self.max_nof_bins = max_nof_bins
        self.min_points_per_bin = min_points_per_bin
        self.discount = discount


class Greedy:
    def __init__(self,
                 max_nof_bins: int = 100,
                 min_points_per_bin: int = 10,
                 fact: float = 1.05,
                 ):
        self.max_nof_bins = max_nof_bins
        self.min_points_per_bin = min_points_per_bin
        self.fact = fact


class Fixed:
    def __init__(self, nof_bins: int = 100, min_points_per_bin=None):
        self.nof_bins = nof_bins
        self.min_points_per_bin = min_points_per_bin
