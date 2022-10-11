from autofit.src.composite_function import CompositeFunction
from autofit.src.primitive_function import PrimitiveFunction

def test_composite_functions():
    test_comp = CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                  branch_list=[PrimitiveFunction.built_in("pow0"),
                                                 PrimitiveFunction.built_in("pow1")],
                                  name="TestMeNow")
    test_comp.print_tree()
    value = test_comp.eval_at(1)
    print(value)

    test_comp2 = CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                  branch_list=[test_comp,test_comp,test_comp],
                                  name="TestMeNowBaby")
    test_comp2.print_tree()
    value2 = test_comp2.eval_at(1)
    print(value2)

    test_comp3 = CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                  branch_list=[test_comp2,test_comp,test_comp2],
                                  name="TestMeNowBabyOoh")
    test_comp3.print_tree()
    value3 = test_comp3.eval_at(1)
    print(value3)