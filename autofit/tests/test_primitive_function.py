"""Tests MIW's AutoFit PrimitiveFunction class"""

# required built-ins

# required external libraries
import numpy as np

# required internal classes
from autofit.src.primitive_function import PrimitiveFunction
from autofit.tests.conftest import assert_almost_equal

# fmt: off
xsuite = [
    -100, -10, -0.1, -0.001, -1e-8, 1e-8, 0.001, 0.1, 10, 100,
    -100j, -10j, -0.1j, -0.001j, -1e-8j, 1e-8j, 0.001j, 0.1j, 10j, 100j,
]
# fmt: on


def test_prim_init():

    # default constructor
    default_prim = PrimitiveFunction()
    assert default_prim.name == "pow1"
    assert repr(default_prim) == "Prim pow1 uses pow1(x,arg) with coefficient 1.0"
    assert default_prim._callable_1param is None
    assert default_prim.func is PrimitiveFunction.pow1
    assert_almost_equal(default_prim.eval_at(0.1), 0.1)
    assert_almost_equal(default_prim.eval_deriv_at(0.1), 1.0)

    # with name and function specified
    test_prim = PrimitiveFunction(name="test_func", func=PrimitiveFunction.my_cos)
    assert test_prim.name == "test_func"
    assert test_prim._callable_1param is None
    assert test_prim.func is PrimitiveFunction.my_cos
    assert_almost_equal(test_prim.eval_at(0.1), np.cos(0.1))
    assert_almost_equal(test_prim.eval_deriv_at(0.1), -np.sin(0.1))

    # with name and other callable specified
    test_other = PrimitiveFunction(name="test_other", other_callable=np.exp)
    assert test_other.name == "test_other"
    assert test_other._callable_1param is np.exp
    # id1 = hex(id(test_other.func))
    # id2 = hex(id(test_other.callable_2param))
    # assert id1 == id2
    assert test_other.func.__name__ is test_other.callable_2param.__name__
    assert_almost_equal(test_other.eval_at(0.1), np.exp(0.1))
    assert_almost_equal(test_other.eval_deriv_at(0.1), np.exp(0.1))


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

    test_shift = PrimitiveFunction.built_in("pow1_shift")

    test_sum = PrimitiveFunction.built_in("sum")

    for xval in xsuite:
        assert_almost_equal(test_pow0.eval_at(xval), np.power(xval, 0))
        assert_almost_equal(test_pow1.eval_at(xval), np.power(xval, 1))
        assert_almost_equal(test_pow2.eval_at(xval), np.power(xval, 2))
        assert_almost_equal(test_pow3.eval_at(xval), np.power(xval, 3))
        assert_almost_equal(test_pow4.eval_at(xval), np.power(xval, 4))
        assert_almost_equal(test_pow_neg1.eval_at(xval), np.float_power(xval, -1))

        assert_almost_equal(test_cos.eval_at(xval), np.cos(xval))
        assert_almost_equal(test_sin.eval_at(xval), np.sin(xval))
        assert_almost_equal(test_exp.eval_at(xval), np.exp(xval))
        assert_almost_equal(test_log.eval_at(xval), np.log(xval))

        assert_almost_equal(test_shift.eval_at(xval), xval - 1)

        assert_almost_equal(test_sum.eval_at(xval), xval)


def test_arbitrary_powers():

    pow2_prim = PrimitiveFunction.built_in("Pow2")
    assert_almost_equal(pow2_prim.eval_at(0.1), 0.1**2)

    pow10_prim = PrimitiveFunction.built_in("Pow10")
    assert_almost_equal(pow10_prim.eval_at(0.9), 0.9**10)


def test_composite_helpers():

    for xval in xsuite:

        assert_almost_equal(
            PrimitiveFunction.dim0_pow2(xval, 0.1), -(xval**2) / (2 * 0.1**2)
        )
        assert_almost_equal(PrimitiveFunction.exp_dim1(xval, 0.1), np.exp(-xval / 0.1))
        assert_almost_equal(
            PrimitiveFunction.n_exp_dim2(xval, 0.1),
            np.exp(-(xval**2) / (2 * 0.1**2)) / np.sqrt(2 * np.pi * 0.1**2),
        )
