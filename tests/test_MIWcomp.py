
# required built-ins

# required internal classes
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.MIWcomp import MIWcomp

# required external libraries
import numpy as np

# Testing tools

def assertRelativelyEqual(exp1, exp2):
    diff = exp1 - exp2
    av = (exp1 + exp2) / 2
    relDiff = np.abs(diff / av)
    assert relDiff < 1e-6


xsuite = [-100 ,-10 ,-0.1 ,-0.001 ,-1e-8 ,1e-8 ,0.001 ,0.1 ,10 ,100 ,
          -100j,-10j,-0.1j,-0.001j,-1e-8j,1e-8j,0.001j,0.1j,10j,100j]

def test_sum_init():

    default_comp = MIWcomp()
    assert default_comp.__repr__() == "pow1 with 0 terms in its argument"
    assert default_comp.sumlen is 0
    assert default_comp.prim.name == "pow1"
    assert default_comp.prim.func is PrimitiveFunction.pow1
    assert default_comp.n_dof is 0

    sin_comp = MIWcomp(prim_=PrimitiveFunction.built_in("sin"))
    assert sin_comp.__repr__() == "my_sin with 0 terms in its argument"
    assert sin_comp.sumlen is 0
    assert sin_comp.prim.name == "my_sin"
    assert sin_comp.prim.func is PrimitiveFunction.my_sin
    assert sin_comp.name == "my_sin"
    assert sin_comp.n_dof is 0


