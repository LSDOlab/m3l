from dataclasses import dataclass

from m3l.core.m3l_classes import FunctionSpace

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class IDWFunctionSpace(FunctionSpace): # this is a bit of a hack I guess
    name : str
    points : np.ndarray
    order : float
    coefficients_shape : tuple

    def compute_evaluation_map(self, parametric_coordinates:np.ndarray):
        dist = cdist(self.points, parametric_coordinates)
        weights = 1.0/dist**self.order
        weights = weights.T
        weights /= weights.sum(axis=0)
        np.nan_to_num(weights, copy=False, nan=1.) # maybe do another weights /= weights.sum(axis=0) after this
        return weights

    def compute_fitting_map(self, parametric_coordinates:np.ndarray):
        dist = cdist(parametric_coordinates, self.points)
        weights = 1.0/dist**self.order
        weights = weights.T
        weights /= weights.sum(axis=0)
        np.nan_to_num(weights, copy=False, nan=1.)
        return weights
    

@dataclass
class IDWFunctionSpace2(FunctionSpace):
    name : str
    points : np.ndarray
    order : float
    coefficients_shape : tuple

    def compute_evaluation_map(self, parametric_coordinates:np.ndarray):
        dist = cdist(self.points, parametric_coordinates)
        weights = 1.0/dist**self.order
        weights /= weights.sum(axis=0)
        weights = weights.T
        np.nan_to_num(weights, copy=False, nan=1.) # maybe do another weights /= weights.sum(axis=0) after this
        return weights