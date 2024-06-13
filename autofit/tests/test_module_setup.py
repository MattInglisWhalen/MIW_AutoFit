"""Checks to see if the module hierarchy works"""

from autofit import *


def test_optimizer():

    default_opt = Optimizer()
    assert len(default_opt.top5_models) == 0


def test_composite():

    gaussian = CompositeFunction.built_in("Gaussian")
    assert gaussian.name == "Gaussian"
