
# required built-ins

# required internal classes
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


xsuite = [-100 ,-10 ,-0.1 ,-0.001 ,-1e-8 ,1e-8 ,0.001 ,0.1 ,10 ,100 ,
          -100j,-10j,-0.1j,-0.001j,-1e-8j,1e-8j,0.001j,0.1j,10j,100j]

def test_comp_init():

    # default constructor
    default_comp = CompositeFunction()
    hello = default_comp.name
    assert default_comp.name is "pow1"
    assert default_comp.func is PrimitiveFunction.pow1
    assert default_comp.dof is 1
    assertRelativelyEqual(default_comp.eval_at(3.14), 3.14)

    # simple sum
    test_sum = CompositeFunction(branch_list=[PrimitiveFunction.built_in("pow0"),PrimitiveFunction.built_in("pow1")])
    assert test_sum.name is "sum(pow0+pow1)"
    assert test_sum.func is PrimitiveFunction.pow1
    assert test_sum.dof is 2
    assertRelativelyEqual(test_sum.eval_at(0.1), 1 + 0.1 )

    # simple product
    test_prod = CompositeFunction(branch_list=[[PrimitiveFunction.built_in("pow2"),PrimitiveFunction.built_in("exp")]])
    assert test_prod.name is "prod(my_exp·pow2)"
    assert test_prod.func is PrimitiveFunction.pow1
    assert test_prod.dof is 1
    assertRelativelyEqual(test_prod.eval_at(0.1), 0.1*0.1*np.exp(0.1) )


    # constructor with no multiplication
    test_comp = CompositeFunction(func=PrimitiveFunction.built_in("cos"),
                                  branch_list=[PrimitiveFunction.built_in("pow0"),PrimitiveFunction.built_in("pow1")])

    assert test_comp.name is "my_cos(pow0+pow1)"
    assert test_comp.func is PrimitiveFunction.built_in("cos")
    assert test_comp.dof is 3
    assertRelativelyEqual(test_comp.eval_at(0.1), np.cos( 1 + 0.1) )



