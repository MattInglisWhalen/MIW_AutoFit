from __future__ import annotations

# built-in libraries
from math import floor
from typing import Callable, Union
import re as regex


# external libraries
import numpy as np

# internal classes
from autofit.src.primitive_function import PrimitiveFunction
import autofit.src.MIWprod
import autofit.src.MIWsum

class MIWcomp:

    """
    A wrapper for a function which takes as input a sum of products of other MIWcomps
    """

    def __init__(self, prim_: PrimitiveFunction = None, sum_: MIWsum = None):

        if prim_ is None :
            self._prim : PrimitiveFunction = PrimitiveFunction.built_in("pow1")
        else :
            self._prim : PrimitiveFunction = prim_

        if sum_ is None :
            self._sum : MIWsum = None
        else:
            self._sum : MIWsum = sum_

    def __repr__(self):
        return f"{self._prim.name} with {self.sumlen} terms in its argument"

    @property
    def prim(self):
        return self._prim
    @property
    def sum(self):
        return self._sum
    @property
    def sumlen(self):
        if self._sum is None:
            return 0
        return len(self._sum)

    @property
    def n_dof(self):
        if self._sum is None:
            return 0
        return self._sum.n_dof
    @property
    def name(self):
        if self._sum is None:
            return self._prim.name
        return f"{self._prim.name}({self._sum.name})"

