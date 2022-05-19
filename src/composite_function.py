

# built-in libraries
import math
import random as rng
from dataclasses import field
from math import floor

# external libraries
import numpy as np

# internal classes
from autofit.src.datum1D import Datum1D
from autofit.src.primitive_function import PrimitiveFunction
import autofit.src.primitive_function as prim

class CompositeFunction:


    """
    A composite function is represented as a tree.
          f
       /    \
      5     13
     /\      \
    2  3     11
    Leaves sharing the same parent node are summed together, and are used as input
    for functional composition in the parent node.
    E.g. f(x) = f5( f2(x) + f3(x) ) + f13( f11(x) )
    """
    """
    The tree for the sigmoid function 1/(1+e^(-x)) is then
                f(x)
                /
           neg_pow1(x,1)
           /        \
      my_exp(x,1)  pow0(x,1)
        /
    pow1(x,-1)
    """

    """
    This class represents the nodes of the tree. A node with no children is a primitive function, while
    a node with no parent is the function tree as a whole.
    """


    def __init__(self, children_list = None, parent = None, func: PrimitiveFunction = None, name = ""):
        self._name = name

        self._parent = None
        if parent is not None :
            self._parent = parent

        self._func_of_this_node = func.copy()
        self._children_list = []
        self._contraints = []  # list of (idx1, func, idx3) triplets, with the interpretation
                               # that par[idx1] = func( par[idx2 )]
        if children_list is not None :
            for child in children_list :
                self.add_child(child)
        if name == "" :
            self.build_name()
        self.calculate_degrees_of_freedom()

    def __repr__(self):
        return self._name

    """
    For __init__
    """

    def add_child(self, child, update_name = True):
        new_child = child.copy()
        if isinstance(new_child, PrimitiveFunction):  # have to convert it to a Composite
            prim_as_comp = CompositeFunction(func=new_child, parent=self)
            self._children_list.append(prim_as_comp)
        else :
            new_child._parent = self
            self._children_list.append(new_child)
        if update_name:
            self.build_name()

    def num_nodes(self):
        nodes = 1  # for self
        for child in self._children_list:
            nodes += child.num_nodes()

        return nodes

    def get_node_with_index(self, n):

        if n == 0 :
            return self

        it = 1
        for child in self._children_list :
            if it <= n < it + child.num_nodes() :
                return child.get_node_with_index(n-it)
            it += child.num_nodes()



    def list_extend_with_added_prim_to_self_and_descendents(self, the_list, prim):

        new_comp_base = self.copy()
        new_comp_base.add_child(prim)
        the_list.append( new_comp_base )

        for ichild in self._children_list :
            new_comp = self.copy()
            new_comp.list_extend_with_added_prim_to_self_and_descendents(the_list=the_list, prim=prim)

    def build_name(self):
        name_str = ""
        name_str += f"{self._func_of_this_node.name}("

        names = sorted([child.name for child in self._children_list])
        for name in names:
            name_str += f"{name}+"
        if len(self._children_list) > 0:
            name_str = name_str[:-1] + "))"
        self._name = name_str[:-1]
        if self._parent is not None:
            self._parent.build_name()

    def add_constraint(self, constraint_3tuple):
        self._contraints.append(constraint_3tuple)

    def calculate_degrees_of_freedom(self):
        num_dof = 1
        for child in self._children_list:
            num_dof += child.calculate_degrees_of_freedom()

        if self._name[0:3] == "pow" and len(self._children_list) > 0:
            # powers are special: they lose a degree of freedom if they have children
            # e.g. A*( B*cos(x) )^2 == A*B^2*cos^2(x) == C*cos^2(x). We keep the outer one as free,
            # and the first parameter in the composition is set to unity
            num_dof -= 1

        return num_dof - len(self._contraints)

    def tree_as_string(self, buffer_chars=0):
        tree_str = f"{self._func_of_this_node.name[:10] : <10}"  # pads and truncates to ensure a length of 10
        for idx, child in enumerate(self._children_list) :
            if idx > 0 :
                tree_str += " " * 10
                for n in range( floor(buffer_chars/10) ) :
                    tree_str += "   " + " " * 10
            tree_str += " ~ "
            tree_str += child.tree_as_string(buffer_chars=buffer_chars+10)
            tree_str += "\n"
        return tree_str[:-2]

    # doesn't work perfectly wih negative coefficients
    def tree_as_string_with_args(self, buffer_chars=0):
        tree_str = f"{self._func_of_this_node.arg:.2E}{self._func_of_this_node.name[:10] : <10}"
        for idx, child in enumerate(self._children_list) :
            if idx > 0 :
                tree_str += " " * 18
                for n in range( floor(buffer_chars/18) ) :
                    tree_str += "   " + " " * 18
            tree_str += " ~ "
            tree_str += child.tree_as_string_with_args(buffer_chars=buffer_chars+18)
            tree_str += "\n"
        return tree_str
        # [:-2]

    def print_tree(self):
        print(f"{self._name}:")
        # print(self.tree_as_string())
        print(self.tree_as_string_with_args())

    def copy(self):
        new_comp = CompositeFunction(name=self._name, parent=None, func=self.func)
        for child in self._children_list :
            new_comp.add_child( child )
        return new_comp

    def has_double_trigness(self):
        if "sin" in self.func.name or "cos" in self.func.name :
            if self.has_trig_children() :
                return True
        for child in self._children_list :
            if child.has_double_trigness():
                return True
        return False

    def has_double_expness(self):
        if "exp" in self.func.name :
            if self.has_exp_children() :
                return True
        for child in self._children_list :
            if child.has_double_expness():
                return True
        return False

    def has_double_logness(self):
        if "log" in self.func.name :
            if self.has_log_children() :
                return True
        for child in self._children_list :
            if child.has_double_logness():
                return True
        return False

    def has_trig_children(self):
        for child in self._children_list :
            if "sin" in child.func.name or "cos" in child.func.name or child.has_trig_children() :
                return True
        return False

    def has_exp_children(self):
        for child in self._children_list :
            if "exp" in child.func.name or child.has_exp_children() :
                return True
        return False

    def has_log_children(self):
        for child in self._children_list :
            if "log" in child.func.name or child.has_log_children() :
                return True
        return False

    def num_children(self):
        return len(self._children_list)



    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, val):
        self._name = val

    @property
    def func(self):
        return self._func_of_this_node

    @property
    def dof(self):
        return self.calculate_degrees_of_freedom()

    def eval_at(self,x):
        children_eval_to = 0
        if len(self._children_list) == 0 :
            return self._func_of_this_node.eval_at(x)
        for child in self._children_list :
            children_eval_to += child.eval_at(x)
        return self._func_of_this_node.eval_at( children_eval_to )

    def eval_deriv_at(self,x):
        # simple symmetric difference. If exact derivative is needed, can add that later
        delta = 1e-5
        # return (self.eval_at(x + delta) - self.eval_at(x)) / delta
        return (self.eval_at(x + delta) - self.eval_at(x - delta)) / (2 * delta)
        # can do higher differences later? https://en.wikipedia.org/wiki/Finite_difference_coefficient
        # return ( self.eval_at(x-2*delta) - 8*self.eval_at(x-delta)
        #           + 8*self.eval_at(x+delta) - self.eval_at(x+2*delta) ) / (12*delta)


    def set_args(self, *args):

        it = 0

        args_as_list = list(args)
        # insert zeroes where the constrained arguments go
        for idx_constrained, _, _ in sorted(self._contraints, key=lambda tup: tup[0]) :
            args_as_list.insert( idx_constrained, 0 )
        # apply the constraints
        for idx_constrained, func, idx_other in sorted(self._contraints, key=lambda tup: tup[0]) :
            args_as_list[idx_constrained] = func( args_as_list[idx_other] )

        self._func_of_this_node.arg = args_as_list[it]
        it += 1
        if self._func_of_this_node.name[0:3] == "pow" and len( self._children_list ) > 0 :
            args_as_list.insert(it,1)

        for child in self._children_list :
            next_dof = child.dof
            child.set_args( *args_as_list[it:it+next_dof] )
            it += next_dof

    def get_args(self):
        all_args = [self._func_of_this_node.arg]
        skip_flag = 0
        if self._func_of_this_node.name[0:3] == "pow" and len( self._children_list ) > 0 :
            skip_flag = 1

        print(self._children_list)
        for child in self._children_list[skip_flag:] :
            all_args.extend( child.get_args() )
        return all_args

    def scipy_func(self, x, *args):
        self.set_args(*args)
        return self.eval_at(x)

    @staticmethod
    def built_in_dict():

        built_ins = {}

        # linear model: m, b as parameters (roughly)
        linear = CompositeFunction(func=PrimitiveFunction.built_in("pow1"),
                                   children_list=[PrimitiveFunction.built_in("pow1"),
                                   PrimitiveFunction.built_in("pow0")]
                                   )

        # Gaussian: A, mu, sigma as parameters (roughly)
        gaussian_inner_quadratic = CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                                     children_list=[PrimitiveFunction.built_in("pow1"),
                                                                    PrimitiveFunction.built_in("pow0")]
                                                    )
        gaussian = CompositeFunction(func=PrimitiveFunction.built_in("exp"),
                                     children_list=[gaussian_inner_quadratic],
                                     name="Gaussian")

        # Normal: mu, sigma as parameters (roughly)
        normal = gaussian.copy()
        normal.add_constraint( (0,gaussian_normalization_constraint,1) )

        # make dict entries
        built_ins["Linear"] = linear
        built_ins["Gaussian"] = gaussian
        built_ins["Normal"] = normal

        return built_ins

    @staticmethod
    def built_in_list():
        built_ins = []
        for key, comp in CompositeFunction.built_in_dict().items():
            built_ins.append(comp)
        return built_ins

    @staticmethod
    def built_in(key):
        return CompositeFunction.built_in_dict()[key]

def sameness_constraint(x):
    return x
def gaussian_normalization_constraint(x):  # A exp[ B(x+C)^2 ] requires A=sqrt(1 / 2 pi sigma^2) and B= - 1 / 2 sigma^2
    return math.sqrt(-x/math.pi)


def test_composite_functions():

    test_comp = CompositeFunction(func=PrimitiveFunction.built_in("pow1"),
                                  children_list=[PrimitiveFunction.built_in("pow0"),
                                                 PrimitiveFunction.built_in("pow1")],
                                  name="TestMeNow")
    test_comp.set_args(5,7)

    print( "pp",test_comp.get_args() )
    test_comp.print_tree()
    value = test_comp.eval_at(1)
    print(value)

    test_comp2 = CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                    children_list=[test_comp,test_comp,test_comp],
                                    name="TestMeNowBaby")
    test_comp2.print_tree()
    value2 = test_comp2.eval_at(1)
    print(value2)

    constraint1 = (0,sameness_constraint,1)
    test_comp.add_constraint(constraint1)
    test_comp.set_args(5)

    test_comp.print_tree()
    value3=test_comp.eval_at(5)
    print(value3)



if __name__ == "__main__":
    test_composite_functions()