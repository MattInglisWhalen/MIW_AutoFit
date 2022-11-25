
# required built-ins

# required internal classes
from autofit.src.MIWsum import MIWsum
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

    default_sum = MIWsum()
    assert default_sum.__repr__() == "Sum with 0 terms"
    assert len(default_sum.prods) is 0
    assert len(default_sum.coeffs) is 0
    assert default_sum.n_dof is 0

    weird_sum = MIWsum(coeffs_=[1])
    assert weird_sum.__repr__() == "Sum with 1 terms"
    assert len(weird_sum.prods) is 1
    assert len(weird_sum.coeffs) is 1
    assert weird_sum.n_dof is 1


    sin_comp = MIWcomp(prim_=PrimitiveFunction.built_in("sin"))
    sin_prod = MIWprod(comps_=[sin_comp])
    sin_sum = MIWsum(prods_=[sin_prod])
    assert sin_sum.__repr__() == "Sum with 1 terms"
    assert len(sin_sum) is 1
    assert len(sin_sum[0]) is 1
    assert sin_sum[0][0].prim.name == "my_sin"
    assert sin_sum[0][0].prim.func is PrimitiveFunction.my_sin
    assert sin_sum.name == "my_sin"
    assert sin_sum.n_dof is 1


