# required built-ins

# required internal classes
import pytest

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


xsuite = [-100 ,-10 ,-0.1 ,-0.001 ,-1e-8 ,1e-8 ,0.001 ,0.1 ,10 ,100 ,
          -100j,-10j,-0.1j,-0.001j,-1e-8j,1e-8j,0.001j,0.1j,10j,100j]

@pytest.fixture
def deep_comp():
    first_level = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                    children_list=[PrimitiveFunction.built_in("pow0"),
                                                   PrimitiveFunction.built_in("pow1")])
    second_level = CompositeFunction(prim_=PrimitiveFunction.built_in("cos"),
                                     children_list=[first_level,first_level,first_level])
    third_level = CompositeFunction(prim_=PrimitiveFunction.built_in("exp"),
                                    children_list=[second_level,second_level])
    return third_level

def test_comp_init():

    assert 1 is 1

    # default constructor
    test_default = CompositeFunction()
    assert len(test_default.children_list) is 0
    assert test_default.younger_brother is None
    assert test_default.older_brother is None
    assert test_default.parent is None
    assert test_default.prim.func is PrimitiveFunction.pow1
    assert test_default.name == "pow1"
    assert len(test_default.constraints) is 0
    assert test_default.dof is 1
    assertRelativelyEqual(test_default.eval_at(3.14), 3.14)

    # composite with children
    test_children = CompositeFunction(children_list=[PrimitiveFunction.built_in("pow0"),
                                                     PrimitiveFunction.built_in("pow1")])
    assert len(test_children.children_list) is 2
    assert test_children.younger_brother is None
    assert test_children.older_brother is None
    assert test_children.parent is None
    assert test_children.children_list[0].parent is test_children
    assert test_children.children_list[1].parent is test_children
    assert test_children.prim.func is PrimitiveFunction.pow1
    assert test_children.children_list[0].prim.func is PrimitiveFunction.pow0
    assert test_children.children_list[1].prim.func is PrimitiveFunction.pow1
    assert test_children.name == "pow1(pow0+pow1)"
    assert len(test_children.constraints) is 0
    assert test_children.dof is 2
    assertRelativelyEqual(test_children.eval_at(0.1), 1 + 0.1 )

    # composite with younger brother
    test_brother = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                     younger_brother=CompositeFunction.built_in("Linear"))
    assert len(test_brother.children_list) is 0
    assert test_brother.younger_brother is not None
    assert test_brother.older_brother is None
    assert test_brother.younger_brother.older_brother is test_brother
    assert test_brother.parent is None
    assert test_brother.younger_brother.parent is None
    assert test_brother.prim.func is PrimitiveFunction.my_sin
    assert test_brother.younger_brother.prim.func is PrimitiveFunction.pow1
    assert test_brother.younger_brother.children_list[0].prim.func is PrimitiveFunction.pow1
    assert test_brother.younger_brother.children_list[1].prim.func is PrimitiveFunction.pow0
    assert test_brother.name == "my_sin·pow1(pow0+pow1)"
    assert len(test_brother.constraints) is 0
    assert test_brother.dof is 2
    assertRelativelyEqual(test_brother.eval_at(0.1), np.sin(0.1)*(1+0.1))

def test_set_get_args():

    test_default = CompositeFunction()
    test_default.set_args(*[3])
    assertRelativelyEqual(test_default.eval_at(0.1), 3*0.1)
    assert all([ a == b for a,b in zip(test_default.get_args(),[3]) ])

    test_children = CompositeFunction(children_list=[PrimitiveFunction.built_in("pow0"),
                                                     PrimitiveFunction.built_in("pow1")])
    test_children.set_args(*[5,7])
    assertRelativelyEqual(test_children.eval_at(0.1), 5*(1+7*0.1))
    assertListEqual(test_children.get_args(), [5,7])

    # composite with younger brother
    test_brother = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                     younger_brother=CompositeFunction.built_in("Linear"))
    test_brother.set_args(*[11,13])
    assertRelativelyEqual(test_brother.eval_at(0.1), 11*np.sin(0.1)*(1*0.1+13))
    assertListEqual(test_brother.get_args(), [11,13])

    # deep conposition with multiplication and summs
    test_deep_mul = CompositeFunction(prim_=PrimitiveFunction.built_in("log"),
                                      younger_brother=test_brother,
                                      children_list=[test_brother,test_brother])
    test_deep_mul.set_args(*[3, 5, 7, 11, 13, 17])
    test_deep_mul_tree_args = test_deep_mul.tree_as_string_with_args()
    assertListEqual(test_deep_mul.get_args(), [3,5,7,11,13,17])
    expected_deep_tree_args = "   +3.00E+00my_log     ~ +5.00E+00my_sin    \n"                       \
                              " |                     x +1.00E+00pow1       ~ +1.00E+00pow1      \n" \
                              " |                     |                     ~ +7.00E+00pow0      \n" \
                              " |                     ~ +1.10E+01my_sin    \n"                       \
                              " |                     x +1.00E+00pow1       ~ +1.00E+00pow1      \n" \
                              " |                                           ~ +1.30E+01pow0      \n" \
                              " x +1.00E+00my_sin    \n"                                             \
                              " x +1.00E+00pow1       ~ +1.00E+00pow1      \n"                       \
                              " |                     ~ +1.70E+01pow0      "
    assert test_deep_mul_tree_args == expected_deep_tree_args

def test_dims():

    # composite with younger brother
    test_brother = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                     younger_brother=CompositeFunction.built_in("Linear"))
    # deep conposition with multiplication and summs
    test_deep_mul = CompositeFunction(prim_=PrimitiveFunction.built_in("log"),
                                      younger_brother=test_brother,
                                      children_list=[test_brother,test_brother])
    test_deep_mul_tree_dims = test_deep_mul.tree_as_string_with_dimensions()
    expected_deep_tree_dims = "   -1/+0my_log     ~ -1/+0my_sin    \n"                   \
                              " |                 x +0/+1pow1       ~ +0/+1pow1      \n" \
                              " |                 |                 ~ +1/+0pow0      \n" \
                              " |                 ~ -1/+0my_sin    \n"                   \
                              " |                 x +0/+1pow1       ~ +0/+1pow1      \n" \
                              " |                                   ~ +1/+0pow0      \n" \
                              " x +0/+0my_sin    \n"                                     \
                              " x +0/+1pow1       ~ +0/+1pow1      \n"                   \
                              " |                 ~ +1/+0pow0      "
    assert test_deep_mul_tree_dims == expected_deep_tree_dims


def test_manip_with_constraints():
    normal = CompositeFunction.built_in("Normal")
    with pytest.raises(AttributeError):
        normal.add_child(PrimitiveFunction.built_in("pow0"))
    with pytest.raises(AttributeError):
        normal.add_younger_brother(PrimitiveFunction.built_in("pow0"))

def test_tree_printout():

    gaussian = CompositeFunction.built_in("Gaussian")
    gaussian_tree = gaussian.tree_as_string()
    expected_gaussian_tree = "my_exp     ~ pow2_force ~ pow1      \n" \
                             "                        ~ pow0      "
    assert gaussian_tree == expected_gaussian_tree

    # composite with younger brother
    test_brother = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                     younger_brother=CompositeFunction.built_in("Linear"))
    test_brother_tree = test_brother.tree_as_string()
    expected_brother_tree = "   my_sin    \n"                \
                            " x pow1       ~ pow1      \n"   \
                            " |            ~ pow0      "
    assert test_brother_tree == expected_brother_tree

    test_deep_mul = CompositeFunction(prim_=PrimitiveFunction.built_in("log"),
                                      younger_brother=test_brother,
                                      children_list=[test_brother,test_brother])
    test_deep_tree = test_deep_mul.tree_as_string()
    expected_deep_tree = "   my_log     ~ my_sin    \n"              \
                         " |            x pow1       ~ pow1      \n" \
                         " |            |            ~ pow0      \n" \
                         " |            ~ my_sin    \n"              \
                         " |            x pow1       ~ pow1      \n" \
                         " |                         ~ pow0      \n" \
                         " x my_sin    \n" \
                         " x pow1       ~ pow1      \n" \
                         " |            ~ pow0      "

    assert test_deep_tree == expected_deep_tree

def test_polynomial_creation() :

    poly3 = CompositeFunction.built_in("Polynomial3")
    assert poly3.name == "pow1(Pow0+Pow1+Pow2+Pow3)"
    assertRelativelyEqual( poly3.eval_at(0.1), (0.1**3 + 0.1**2 + 0.1 + 1))

    poly10 = CompositeFunction.built_in("Polynomial10")
    poly10.set_args(*[3,4,5,6,7,8,9,10,11,12,13])
    assert poly10.name == "pow1(Pow0+Pow1+Pow10+Pow2+Pow3+Pow4+Pow5+Pow6+Pow7+Pow8+Pow9)"
    assertRelativelyEqual( poly10.eval_at(0.9), 3*(0.9**10 + 4*0.9**9 + 5*0.9**8 + 6*0.9**7 + 7*0.9**6 + 8*0.9**5 +
                                                   + 9*0.9**4  + 10*0.9**3 + 11*0.9**2 + 12*0.9 + 13 ))

def test_new_stuff():

    pass