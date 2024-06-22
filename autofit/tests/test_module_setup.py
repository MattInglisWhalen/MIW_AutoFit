"""Checks to see if the module hierarchy works"""

# built-in libraries

# external packages

# interna classes
from autofit import *
from autofit.tests.conftest import tkinter_available


def test_composite_loaded():

    gaussian = CompositeFunction.built_in("Gaussian")
    assert gaussian.name == "Gaussian"


def test_datum1D_loaded():

    new_point = Datum1D(pos=0.1, val=2)
    assert new_point.pos == 0.1
    assert new_point.val == 2


def test_data_handler_loaded():

    new_handler = DataHandler(filepath=pkg_path() + "/data/linear_data_yerrors.csv")
    assert new_handler.shortpath == "linear_data_yerrors.csv"


def test_optimizer_loaded():

    default_opt = Optimizer()
    assert len(default_opt.top5_models) == 0


@tkinter_available
def test_frontend_loaded():

    gui = Frontend()
    gui.shutdown()


def test_package_utils_loaded():
    logger("test_module_setup: print with logger")
    debug("test_module_setup: this message should be in red")

    @performance
    def quick_func(x: float):
        return x * x

    quick_func(2)


def test_primitive_loaded():
    default_prim = PrimitiveFunction()
    assert default_prim.name == "pow1"
