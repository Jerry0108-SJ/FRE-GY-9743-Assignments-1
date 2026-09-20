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
        n = self.length_
        # left side, flat
        if x <= self.axis1_[0]:
            return float(self.values_[0])
        # right side, flat
        if x > self.axis1_[n - 1]:
            return float(self.values_[n - 1])
        # inside: find i with axis1[i-1] < x <= axis1[i], value is values[i]
        for i in range(1, n):
            if x <= self.axis1_[i]:
                return float(self.values_[i])

    def integrate(self, start_x: float, end_x: float) -> float:
        # if the bounds are reversed, swap them and flip the sign
        sign = 1.0
        if start_x > end_x:
            start_x, end_x = end_x, start_x
            sign = -1.0

        n = self.length_
        total = 0.0

        # part on the left of the first node
        if start_x < self.axis1_[0]:
            right = min(end_x, self.axis1_[0])
            total += (right - start_x) * self.values_[0]

        # parts between the nodes, interval (axis1[i-1], axis1[i]] uses values[i]
        for i in range(1, n):
            left = max(start_x, self.axis1_[i - 1])
            right = min(end_x, self.axis1_[i])
            if right > left:
                total += (right - left) * self.values_[i]

        # part on the right of the last node
        if end_x > self.axis1_[n - 1]:
            left = max(start_x, self.axis1_[n - 1])
            total += (end_x - left) * self.values_[n - 1]

        return sign * total

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        # f(x) is just one of the y's, so the gradient is 1 for that y and 0 for the others
        n = self.length_
        grad = np.zeros(n)
        if x <= self.axis1_[0]:
            grad[0] = 1.0
        elif x > self.axis1_[n - 1]:
            grad[n - 1] = 1.0
        else:
            for i in range(1, n):
                if x <= self.axis1_[i]:
                    grad[i] = 1.0
                    break
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        # integral = sum of (length of piece) * y_i, so d/dy_i is the length of the piece
        # same loop as integrate(), but store the lengths instead of adding length * value
        sign = 1.0
        if start_x > end_x:
            start_x, end_x = end_x, start_x
            sign = -1.0

        n = self.length_
        grad = np.zeros(n)

        if start_x < self.axis1_[0]:
            right = min(end_x, self.axis1_[0])
            grad[0] += right - start_x

        for i in range(1, n):
            left = max(start_x, self.axis1_[i - 1])
            right = min(end_x, self.axis1_[i])
            if right > left:
                grad[i] += right - left

        if end_x > self.axis1_[n - 1]:
            left = max(start_x, self.axis1_[n - 1])
            grad[n - 1] += end_x - left

        return sign * grad


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