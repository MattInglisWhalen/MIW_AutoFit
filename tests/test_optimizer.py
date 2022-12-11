# required built-ins

# required internal classes
import pytest

from autofit.src.optimizer import Optimizer
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.composite_function import CompositeFunction

# required external libraries
import numpy as np

# Testing tools

def assertRelativelyEqual(exp1, exp2):
    diff = exp1 - exp2
    av = (exp1 + exp2) / 2
    relDiff = np.abs(diff / av)
    assert relDiff < 1e-6

def assertListEqual(l1, l2):
    list1 = list(l1)
    list2 = list(l2)
    assert len(list1) == len(list2)
    for a, b in zip(list1, list2) :
        assertRelativelyEqual(a,b)

def test_init():

    default_opt = Optimizer()
