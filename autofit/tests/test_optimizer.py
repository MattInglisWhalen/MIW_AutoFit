"""Tests MIW's AutoFit Optimizer class"""

# built-in libraries

# external packages

# internal classes
from autofit.src.optimizer import Optimizer
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.composite_function import CompositeFunction


def test_default_constructor():

    default_opt = Optimizer()
    assert len(default_opt.top5_models) == 0


def test_all_loaded():

    all_functions_dict = {
        "cos(x)": True,
        "sin(x)": True,
        "exp(x)": True,
        "log(x)": True,
        "1/x": True,
    }
    opt = Optimizer(use_functions_dict=all_functions_dict, max_functions=3)
    assert len(opt.composite_function_list) == 0
    opt.build_composite_function_list()
    assert len(opt.composite_function_list) == 116
