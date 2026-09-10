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
    """Abstract interface for a 1-D interpolator.

    Do NOT modify this class. Implement the abstract methods in a subclass.
    """

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

    ============================ CONVENTION ============================
    Let axis1 = [a_0, ..., a_{n-1}] (strictly increasing) and
        values = [v_0, ..., v_{n-1}].

        f(x) = v_0        if  x <= a_0                       (FLAT left  extrapolation)
        f(x) = v_j        if  a_{j-1} < x <= a_j,  j = 1..n-1
        f(x) = v_{n-1}    if  x > a_{n-1}                    (FLAT right extrapolation)

    In words: ordinate v_j owns the bucket that ENDS at its own abscissa a_j,
    and each bucket is open on the left, closed on the right. The value exactly
    AT a node therefore belongs to the bucket ending at that node, so
    f(a_j) = v_j for every node. This is what makes f left-continuous.

    Worked example --- axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:

        f(0.5) = 3      f(1)   = 3      f(1.5) = 4      f(3) = 4
        f(5.5) = 6      f(6.5) = 6      f(8)   = 6

    Note f(1) = 3 and f(3) = 4, NOT 4 and 5. Endpoints are where most
    implementations go wrong -- test them first.
    ====================================================================
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        """Return f(x) under the convention documented in the class docstring."""
        ### TODO
        pass

    def integrate(self, start_x: float, end_x: float) -> float:
        """Return the definite integral of f from start_x to end_x.

        You may assume start_x <= end_x. Both endpoints may lie anywhere,
        including outside [a_0, a_{n-1}] on either side.
        """
        ### TODO
        pass

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        """Return d f(x) / d values, as an array of length n.

        Element k is the partial derivative of f(x) with respect to v_k.
        """
        ### TODO
        pass

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        """Return d (integral from start_x to end_x) / d values, length n.

        Element k is the partial derivative of the integral with respect to v_k.
        """
        ### TODO
        pass


class InterpolatorFactory:
    """Do NOT modify this class."""

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
