
# required built-ins

# required internal classes
from autofit.src.MIWprod import MIWprod
from autofit.src.MIWcomp import MIWcomp

from autofit.src.primitive_function import PrimitiveFunction

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

    default_prod = MIWprod()
    assert default_prod.__repr__() == "Product with 0 factors"
    assert len(default_prod) is 0
    assert default_prod.n_dof is 0

    sin_comp = MIWcomp(prim_=PrimitiveFunction.built_in("sin"))
    sin_prod = MIWprod(comps_=[sin_comp])
    assert sin_prod.__repr__() == "Product with 1 factors"
    assert len(sin_prod) is 1
    assert sin_prod[0].prim.name == "my_sin"
    assert sin_prod[0].prim.func is PrimitiveFunction.my_sin
    assert sin_prod.name == "my_sin"
    assert sin_prod.n_dof is 0


