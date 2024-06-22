"""Tests MIW's AutoFit Optimizer class"""

# built-in libraries

# external packages

# internal classes
from autofit.src.optimizer import Optimizer


def test_default_constructor():

    default_opt = Optimizer()
    assert len(default_opt.top5_models) == 0


def test_all_loaded():

    opt = Optimizer(use_functions_dict=Optimizer.all_defaults_on_dict(), max_functions=3)
    assert len(opt.composite_function_list) == 0
    opt.build_composite_function_list()
    assert len(opt.composite_function_list) == 116
