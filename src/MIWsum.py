from __future__ import annotations

# built-in libraries
from math import floor
from typing import Callable, Union
import re as regex


# external libraries
import numpy as np

# internal classes
import autofit.src.MIWprod
import autofit.src.MIWcomp

class MIWsum:

    """
    The head of a tree-like structure representing a general function.
    """

    """
    We
    """

    def __init__(self, prods_: list[MIWprod] = None, coeffs_: list[float] = None, parent_: MIWcomp = None):

        # don't allow mismatch if products and coeffs are given
        if prods_ is not None and coeffs_ is not None :
            if len(prods_) != len(coeffs_) :
                raise AttributeError

        # A product here is one of the terms in this level of the sum. Deeper convolutions are not included
        self._prods : list[MIWprod] = []
        if prods_ is not None :
            self._prods = prods_
        # A coeff here is one of the prefactors in this level of the sum. Deeper convolutions are not included
        self._coeffs : list[float] = []
        if coeffs_ is not None :
            self._coeffs = coeffs_

        # ensure _prods and _coeffs have the same length
        self.fix_mismatch()

        # The parent convolution of this sum
        self._parent = parent_

    def __repr__(self):
        return f"Sum with {len(self)} terms"
    def __len__(self):
        return len(self._prods)
    def __getitem__(self, idx):
        return self._prods[idx]

    def fix_mismatch(self) -> None:
        for _ in range( len(self._prods) - len(self._coeffs) ):
            self._coeffs.append(1)
        for _ in range( len(self._coeffs) - len(self._prods) ):
            self._prods.append(autofit.src.MIWprod.MIWprod())

    # Most important function
    def is_coeff_idx_a_dof(self, idx):
        if self._parent is None :
            return True
        if self._parent.prim.name[0:3] not in ["pow"] :
            return True
        if idx != 0 :
            return True
        return False

    @property
    def prods(self) -> list[MIWprod]:
        return self._prods
    @prods.setter
    def prods(self, terms):
        if len(terms) != len(self._coeffs):
            raise AttributeError
        self._prods = terms
    @property
    def coeffs(self) -> list[float]:
        return self._coeffs
    @coeffs.setter
    def coeffs(self, vals):
        if len(vals) != len(self._prods):
            raise AttributeError
        self._coeffs = vals
    @property
    def parent(self):
        return self._parent

    def set_prods_args(self, terms, vals):
        if len(terms) != len(vals) :
            raise AttributeError
        self._prods = terms
        self._coeffs = vals

    def args(self) -> list[float]:
        args_list = self._coeffs
        for term in self._prods :
            for factor in term.comps :
                args_list.extend(factor.sum.args())
        return args_list

    @property
    def n_dof(self):
        count = sum([ 1 if self.is_coeff_idx_a_dof(idx) else 0 for idx in range(len(self._coeffs))])
        for term in self._prods :
            count += term.n_dof
        return count
    @property
    def name(self):
        name_str = ""
        for term in self._prods :
            name_str += f"{term.name}+"
        return name_str[0:-1]

