from __future__ import annotations

# built-in libraries
from math import floor
from typing import Callable, Union
import re as regex

# external libraries
import numpy as np

# internal classes
import autofit.src.MIWcomp


class MIWprod:

    """
    The product of compositional functions.
    """

    def __init__(self, comps_: list[MIWcomp] = None):

        self._comps: list[MIWprod] = []  # compositions / MIWcomp
        if comps_ is not None:
            self._comps = comps_

    def __repr__(self) -> str:
        return f"Product with {len(self)} factors"
    def __len__(self) -> int:
        return len(self._comps)
    def __getitem__(self, idx) -> MIWcomp:
        return self._comps[idx]

    @property
    def comps(self):
        return self._comps

    @property
    def n_dof(self):
        count = 0
        for factor in self._comps :
            count += factor.n_dof
        return count
    @property
    def name(self):
        name_str = ""
        for factor in self._comps :
            name_str += f"{factor.name}·"
        return name_str[0:-1]

