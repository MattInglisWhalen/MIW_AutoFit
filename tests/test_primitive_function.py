
# required built-ins

# required internal classes
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

def test_prim_init():

    # default constructor
    default_prim = PrimitiveFunction()
    assert default_prim.name is "pow1"
    assertRelativelyEqual( default_prim.eval_at(0.1), 0.1 )
    assertRelativelyEqual( default_prim.eval_deriv_at(0.1), 1. )

    # with name and function specified
    test_prim = PrimitiveFunction(name="test_func", func=PrimitiveFunction.built_in("cos"))
    assert test_prim.name is "test_func"
    assert test_prim.func is PrimitiveFunction.built_in("cos")


def test_prim_builtins():

    test_pow0 = PrimitiveFunction.built_in("pow0")
    test_pow1 = PrimitiveFunction.built_in("pow1")
    test_pow2 = PrimitiveFunction.built_in("pow2")
    test_pow3 = PrimitiveFunction.built_in("pow3")
    test_pow4 = PrimitiveFunction.built_in("pow4")
    test_pow_neg1 = PrimitiveFunction.built_in("pow_neg1")

    test_cos = PrimitiveFunction.built_in("cos")
    test_sin = PrimitiveFunction.built_in("sin")
    test_exp = PrimitiveFunction.built_in("exp")
    test_log = PrimitiveFunction.built_in("log")

    for xval in xsuite :
        assertRelativelyEqual( test_pow0.eval_at(xval), np.power(xval,0))
        assertRelativelyEqual( test_pow1.eval_at(xval), np.power(xval,1))
        assertRelativelyEqual( test_pow2.eval_at(xval), np.power(xval,2))
        assertRelativelyEqual( test_pow3.eval_at(xval), np.power(xval,3))
        assertRelativelyEqual( test_pow4.eval_at(xval), np.power(xval,4))
        assertRelativelyEqual( test_pow_neg1.eval_at(xval), np.float_power(xval,-1))

        assertRelativelyEqual( test_cos.eval_at(xval), np.cos(xval) )
        assertRelativelyEqual( test_sin.eval_at(xval), np.sin(xval) )
        assertRelativelyEqual( test_exp.eval_at(xval), np.exp(xval) )
        assertRelativelyEqual( test_log.eval_at(xval), np.log(xval*xval)/2 )








