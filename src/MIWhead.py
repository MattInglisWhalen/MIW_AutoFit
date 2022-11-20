from __future__ import annotations

# built-in libraries
from math import floor
from typing import Callable, Union
import re as regex

# external libraries
import numpy as np

# internal classes
from autofit.src.MIWprod import MIWprod


class MIWsum:
    """

    An MIWsum is the head of a tree-like structure representing a general function.


    """

    def __init__(self, prods: list[MIWprod] = None, args: list[float] = None):

        self._prods: list[MIWprod] = []
        if prods is not None:
            self._prods = prods
        self._args: list[float] = []
        if args is not None:
            self._args = args

    def __repr__(self):
        if len(self._prods) < 1:
