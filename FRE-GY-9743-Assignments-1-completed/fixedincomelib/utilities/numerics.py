import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        if self.length_ == 0:
            raise ValueError("Cannot interpolate with an empty axis")

        # For the left-continuous convention, y[i] applies on
        # (axis1[i - 1], axis1[i]].  searchsorted(..., side='left')
        # therefore returns exactly the required ordinate index.  Clipping
        # the index supplies flat extrapolation on both wings.
        index = min(int(np.searchsorted(self.axis1_, x, side='left')),
                    self.length_ - 1)
        return self.values_[index]

    def integrate(self, start_x: float, end_x: float) -> float:
        weights = self.gradient_of_integrated_value_wrt_ordinate(
            start_x, end_x)
        return float(np.dot(weights, self.values_))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        if self.length_ == 0:
            raise ValueError("Cannot compute a gradient with an empty axis")

        gradient = np.zeros(self.length_, dtype=float)
        index = min(int(np.searchsorted(self.axis1_, x, side='left')),
                    self.length_ - 1)
        gradient[index] = 1.0
        return gradient

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        if self.length_ == 0:
            raise ValueError("Cannot integrate with an empty axis")
        if start_x == end_x:
            return np.zeros(self.length_, dtype=float)
        if start_x > end_x:
            return -self.gradient_of_integrated_value_wrt_ordinate(
                end_x, start_x)

        # Each entry is the length of the overlap between [start_x, end_x]
        # and the region controlled by the corresponding ordinate.
        if self.length_ == 1:
            return np.array([end_x - start_x], dtype=float)

        gradient = np.zeros(self.length_, dtype=float)

        # y[0] controls the flat left wing through axis1[0].
        gradient[0] = max(0.0, min(end_x, self.axis1_[0]) - start_x)

        # Interior y[i] controls (axis1[i - 1], axis1[i]].
        for i in range(1, self.length_ - 1):
            left = max(start_x, self.axis1_[i - 1])
            right = min(end_x, self.axis1_[i])
            gradient[i] = max(0.0, right - left)

        # The last ordinate controls the final bucket and flat right wing.
        gradient[-1] = max(0.0, end_x - max(start_x, self.axis1_[-2]))
        return gradient


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
