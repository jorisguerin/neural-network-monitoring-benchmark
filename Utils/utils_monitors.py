import numpy as np


class Box:

    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.centers = (self.lower_bounds + self.upper_bounds) / 2
        self.distance_box = ((self.upper_bounds - self.lower_bounds) / 2) + 1e-20

    def is_in_box(self, points):
        is_above_min = np.min(points - self.lower_bounds, axis=1) >= 0
        is_below_max = np.min(self.upper_bounds - points, axis=1) >= 0

        return is_above_min & is_below_max

    def score(self, points):
        distances = np.abs(points - self.centers)

        return np.max(distances / self.distance_box, axis=1)


class Boxes:

    def __init__(self):
        self.boxes = []

    def add_box(self, box):
        self.boxes.append(box)

    def score(self, points):
        all_scores = []
        for b in self.boxes:
            all_scores.append(b.score(points))

        return np.min(all_scores, axis=0)
