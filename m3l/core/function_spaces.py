from dataclasses import dataclass

from m3l.core.m3l_classes import FunctionSpace

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class IDWFunctionSpace(FunctionSpace):
    name : str
    points : np.ndarray
    order : float
    coefficients_shape : tuple

    def compute_evaluation_map(self, parametric_coordinates:np.ndarray):
        dist = cdist(self.points, parametric_coordinates)
        weights = 1.0/dist**self.order
        weights /= weights.sum(axis=0)
        return weights.T


