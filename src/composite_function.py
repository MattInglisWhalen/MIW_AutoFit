from __future__ import annotations

# built-in libraries
from math import floor
from typing import Callable, Union
import re as regex


# external libraries
import numpy as np

# internal classes
from autofit.src.primitive_function import PrimitiveFunction

class CompositeFunction:

    # TODO: think how to structure this so that a sum/pow1 is always at the top-level, but doesn't show in the name
    """
    A composite function is represented as a tree.
            sin
         /     \   \
      1 . 4     6   8
     /\    \     \
    2  3   5      7
    Branches sharing the same parent node are summed together and leaves sharing the same branch
    are multiplied together. These are used as input
    for functional composition in the parent node. E.g., with reference to the above tree
    E.g. f(x) = sin[ f1( f2(x) + f3(x) ) * f4( f5(x) ) + f6( f7(x) ) + f8(x)]
    """
    """
    The tree for the sigmoid function 2/(3+4e^(-5x)) is then
           neg_pow1(x)
           /        \
      my_exp(x)  pow0(x)
        /
    pow1(x)
    
    with args (2,4,-5,3)
    """

    """
    This class represents the nodes of the tree. A node with no children is just a wrapper for a primitive function, 
    while a node with no parent is the function tree as a whole.
    
    Testing coalescence
    """

    """
    We are currently working on implementing multiplication between shared branches. Need to redo dof calcs 
    and get/set arg methods with new paradigm where it is the Composite function which keeps track of the args, not
    the primitive function
    """


    def __init__(self,
                 func: PrimitiveFunction = None,
                 parent : CompositeFunction = None,
                 name: str = "",
                 branch_list = None,
                 args: list[float] = None):

        if branch_list is None :
            branch_list = []

        # the head function which takes as argument the output of all branches
        if func is None :
            self._prim_of_this_node : PrimitiveFunction = PrimitiveFunction.built_in("pow1").copy()
            if name == "" and len(branch_list) > 1:
                name = "sum"
        else:
            self._prim_of_this_node : PrimitiveFunction = func.copy()

        # parent node
        self._parent : CompositeFunction = parent

        # children_list should nominally be a list of terms (branches),
        # with each branch containing a list of factors (leaves)
        self._branch_list : list[list[CompositeFunction]] = []
        if branch_list is not None :
            for branch in branch_list :
                self.add_branch(branch, update_name=False)
            if len(self._branch_list) == 1 and len(self._branch_list[0]) > 1 :
                name = "prod"

        # short name of the function
        if name != "" :
            self._name = name
        else:
            self.build_name()

        # list of (idx1, func, idx2) triplets, with the interpretation that par[idx1] = func( par[idx2] )
        # each _contraint reduces dof by 1
        self._constraints: (int, Callable[[float], float], int) = []
        # in constrast, non-holonomic constraint (_nh_constraint) don't reduce dof
        self._nh_constraints: (int, Callable[[float], float], int) = []

        # the arguments/parameters of each term: one for the head function and one for each term
        if args is None or len(args) != self.calculate_degrees_of_freedom() :
            self._args = [1 for _ in range(self.dof)]
        else :
            self._args = args



    def __repr__(self):
        return f"Composite {self._name} w/ {self.dof} dof"

    """
    For __init__
    """

    def add_branch(self, branch_to_add: Union[CompositeFunction,list[CompositeFunction]], update_name = True):
        # branch could either be a single Composite or a list of Composites -- hooray for duck-typing!
        factors = []
        if isinstance(branch_to_add, list) :
            for leaf in branch_to_add :
                self.add_leaf_to_branch(factors,leaf)
        else:
            self.add_leaf_to_branch(factors,branch_to_add)

        self._branch_list.append(factors)

        if update_name:
            self.build_name()

    def add_leaf_to_branch(self, branch: list[CompositeFunction], leaf: Union[CompositeFunction,PrimitiveFunction]):
        new_leaf = leaf.copy()
        if isinstance(new_leaf, PrimitiveFunction):  # have to convert it to a Composite
            prim_as_comp = CompositeFunction(func=new_leaf, parent=self)
            branch.append(prim_as_comp)
        else :
            new_leaf._parent = self
            branch.append(new_leaf)

    @staticmethod
    def term_name(term: list[CompositeFunction]):

        name_str = ""
        factor_names = sorted([factor.name for factor in term])
        for name in factor_names:
            name_str += f"{name}·"
        return name_str[:-1]

    def build_name(self):

        name_str = ""
        name_str += f"{self._prim_of_this_node.name}("

        term_names = sorted([ CompositeFunction.term_name(term) for term in self._branch_list ])
        for name in term_names:
            name_str += f"{name}+"
        if len(term_names) > 0:
            name_str = name_str[:-1] + "))"
        self._name = name_str[:-1]

        if self._parent is not None:
            self._parent.build_name()

    def add_constraint(self, constraint_3tuple: (int, Callable[[float],float], float)):
        self._constraints.append(constraint_3tuple)
    def add_non_holonomic_constraint(self, constraint_3tuple: (int, Callable[[float],float], float)):
        self._nh_constraints.append(constraint_3tuple)

    """
    Information about this function
    """

    def num_nodes(self):
        nodes = 1  # for self
        for branch in self._branch_list:
            for leaf in branch :
                nodes += leaf.num_nodes()
        return nodes

    def calculate_degrees_of_freedom(self):

        # one for the func_of_this_node and one for each term/branch

        if self._parent is None :
            num_dof = 1
        else:
            num_dof = 0

        for branch in self._branch_list:
            num_dof += 1
            for leaf in branch:
                num_dof += leaf.calculate_degrees_of_freedom()

        if self._name[0:3] in ["pow","sum","pro"] and len(self._branch_list) > 0:
            # powers are special: they lose a degree of freedom if they have children
            # e.g. pow: A( B cos(x) )^2 == A B^2 cos^2(x) == C cos^2(x). We keep the outer one as free,
            # and the first parameter in the composition is set to unity
            num_dof -= 1

        return num_dof - len(self._constraints)

    def tree_as_string(self, buffer_chars=0):
        buff_num = 10
        head_name = self._name.split('(')[0]
        tree_str = f"{head_name[:10] : <10}"  # pads and truncates to ensure a length of 10
        for idx, branch in enumerate(self._branch_list) :
            for jdx, leaf in enumerate(branch) :
                if idx > 0 and jdx > 0 :
                    tree_str += " " * buff_num
                    for n in range( floor(buffer_chars/buff_num) ) :
                        tree_str += "   " + " " * buff_num
                if jdx > 0 :
                    tree_str += " x "
                else :
                    tree_str += " ~ "
                tree_str += leaf.tree_as_string(buffer_chars=buffer_chars+buff_num)
                tree_str += "\n"
        return tree_str[:-2]

    # doesn't work perfectly wih negative coefficients
    def tree_as_string_with_args(self, buffer_chars=0):
        buff_num = 19
        head_name = self._name.split('(')[0]
        tree_str = f"{self.arg[0]:+.2E}{head_name[:10] : <10}"
        for idx, branch in enumerate(self._branch_list) :
            for jdx, leaf in enumerate(branch) :
                if idx > 0 and jdx > 0:
                    tree_str += " " * buff_num
                    for n in range( floor(buffer_chars/buff_num) ) :
                        tree_str += "   " + " " * buff_num
                if jdx > 0 :
                    tree_str += " x "
                else :
                    tree_str += " ~ "
                tree_str += leaf.tree_as_string_with_args(buffer_chars=buffer_chars+buff_num)
                tree_str += "\n"
        return tree_str

    def tree_as_string_with_dimensions(self, buffer_chars=0):
        buff_num = 13
        head_name = self._name.split('(')[0]
        tree_str = f"{self.dimension_arg}/{self.dimension_func}{head_name[:10] : <10}"
        for idx, branch in enumerate(self._branch_list) :
            for jdx, leaf in enumerate(branch) :
                if idx > 0 and jdx > 0:
                    tree_str += " " * buff_num
                    for n in range( floor(buffer_chars/buff_num) ) :
                        tree_str += "   " + " " * buff_num
                if jdx > 0:
                    tree_str += " x "
                else:
                    tree_str += " ~ "
                tree_str += leaf.tree_as_string_with_dimensions(buffer_chars=buffer_chars+buff_num)
                tree_str += "\n"
        return tree_str

    def print_tree(self):
        print(f"Tree {self._name}:")
        # print(self.tree_as_string())
        print(self.tree_as_string_with_args())

    """
    Utility
    """

    def get_node_with_index(self, n):
        if n == 0 :
            return self
        it = 1
        for branch in self._branch_list :
            for leaf in branch :
                if it <= n < it + leaf.num_nodes() :
                    return leaf.get_node_with_index(n-it)
                it += leaf.num_nodes()

    def copy(self):
        new_comp = CompositeFunction(func=self.func.copy(),
                                     parent=None,
                                     name=self.name,
                                     args=self._args.copy())
        for branch in self._branch_list :
            new_comp.add_branch( branch )
        for constraint in self._constraints :
            new_comp.add_constraint( constraint )
        for nh_constraint in self._nh_constraints :
            new_comp.add_non_holonomic_constraint( nh_constraint )
        return new_comp

    # Don't know where this is needed, potentially cuttable
    # def list_extend_with_added_prim_to_self_and_descendents(self, the_list: list[CompositeFunction],
    #                                                               new_prim: PrimitiveFunction):
    #
    #     new_comp_base = self.copy()
    #     new_comp_base.add_child(new_prim)
    #     the_list.append( new_comp_base )
    #
    #     for _ in self._children_list :
    #         new_comp = self.copy()
    #         new_comp.list_extend_with_added_prim_to_self_and_descendents(the_list=the_list, new_prim=new_prim)








    """
    Classifiers for "too much function composition"
    """

    def has_double_trigness(self):
        if "sin" in self.func.name or "cos" in self.func.name :
            if self.has_trig_children() :
                return True
        for child in self.leaf_list :
            if child.has_double_trigness():
                return True
        return False

    def num_trig(self):
        return self.num_cos() + self.num_sin()
    def num_cos(self):
        return self.name.count("my_cos")
    def num_sin(self):
        return self.name.count("my_sin")

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


    """
    Properties
    """

    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, val):
        self._name = val

    @property
    def func(self) -> PrimitiveFunction:
        return self._prim_of_this_node
    @func.setter
    def func(self, other: Callable[[float],float]):
        self._prim_of_this_node = other

    @property
    def dof(self) -> int:
        return self.calculate_degrees_of_freedom()
    @property
    def args(self) -> list[float]:
        return self._args
    @args.setter
    def args(self, args_list) :
        self._args = args_list

    @property
    def parent(self) -> CompositeFunction:
        return self._parent
    @property
    def branch_list(self) -> list[list[CompositeFunction]]:
        return self._branch_list
    @property
    def leaf_list(self) -> list[CompositeFunction]:
        leaves = []
        for branch in self._branch_list :
            leaves.extend([ leaf for leaf in branch])
        return leaves

    """
    Evaluation
    """

    def eval_at(self, x, X0 = 0, Y0 = 0):

        if X0 :
            # print(f"{X0=}")
            # the model is working with LX as the independent variable, but we're being passed x
            LX = np.log(x/X0)
            x = LX

        if len(self._branch_list) == 0 :
            if Y0 :
                LY = self.args[0]*self._prim_of_this_node.eval_at(x)
                y = Y0*np.exp(LY)
            else:
                y = self.args[0]*self._prim_of_this_node.eval_at(x)
            return y
        # else ...

        children_eval_to = 0

        for ibranch, branch in enumerate(self._branch_list) :
            branch_evals_to = self.arg[ibranch+1]
            for leaf in branch :
                branch_evals_to *= leaf.eval_at(x)
            children_eval_to += child.eval_at(x)
        if Y0 :
            # print(f"{Y0=}")
            # the model is working with LY as the dependent variable, but we're expecting to return x
            LY = self._prim_of_this_node.eval_at(children_eval_to)
            y = Y0*np.exp(LY)
        else:
            y = self._prim_of_this_node.eval_at(children_eval_to)
        return y

    def eval_deriv_at(self,x):
        # simple symmetric difference. If exact derivative is needed, can add that later
        delta = 1e-5
        return (self.eval_at(x + delta) - self.eval_at(x - delta)) / (2 * delta)
        # can do higher differences later? https://en.wikipedia.org/wiki/Finite_difference_coefficient
        # return ( self.eval_at(x-2*delta) - 8*self.eval_at(x-delta)
        #           + 8*self.eval_at(x+delta) - self.eval_at(x+2*delta) ) / (12*delta)


    """
    Get and sets for term arguments
    """

    def set_args(self, *args):

        print(f"In {self._name=} set_args to {args=}")
        it = 0

        args_as_list = list(args)
        assert len(args_as_list) is self.dof
        # insert zeroes where the constrained arguments go
        for idx_constrained, _, _ in sorted(self._constraints, key=lambda tup: tup[0]) :
            args_as_list.insert( idx_constrained, 0 )
        # apply the constraints
        for idx_constrained, func, idx_other in sorted(self._constraints, key=lambda tup: tup[0]) :
            args_as_list[idx_constrained] = func( args_as_list[idx_other] )

        try:
            self._prim_of_this_node.arg = args_as_list[it]
        except IndexError:
            # print(f"In set_args {self._name=} {args_as_list=}")
            raise IndexError
        it += 1
        if self._prim_of_this_node.name[0:3] == "pow" and len(self._children_list) > 0 :
            args_as_list.insert(it,1)

        for child in self._children_list :
            next_dof = child.dof
            child.set_args( *args_as_list[it:it+next_dof] )
            it += next_dof

    def get_args(self, skip_flag=0):  # this needs to be massively rewritten if we add multiplication functionality
        # get all arguments normally, then pop off the ones with constraints once we get to the head
        all_args = []
        if skip_flag :
            pass
        else :
            all_args.append(self._prim_of_this_node.arg)

        skip_flag = 0
        if self._prim_of_this_node.name[0:3] == "pow" and len(self._children_list) > 0 :
            skip_flag = 1

        for child in self._children_list :
            all_args.extend( child.get_args(skip_flag) )
            skip_flag = 0

        for idx_constrained, _, _ in sorted(self._constraints, key=lambda tup: tup[0], reverse=True) :
            del all_args[idx_constrained]

        return all_args

    def get_nodes_with_freedom(self, skip_flag=0):
        # gets all nodes which are associated with a degree of freedom
        all_nodes = []
        if skip_flag :
            pass
        else :
            all_nodes.append(self)

        skip_flag = 0
        if self._prim_of_this_node.name[0:3] == "pow" and len(self._children_list) > 0 :
            skip_flag = 1

        for child in self._children_list :
            all_nodes.extend( child.get_nodes_with_freedom(skip_flag) )
            skip_flag = 0

        for idx_constrained, _, _ in sorted(self._constraints, key=lambda tup: tup[0], reverse=True) :
            del all_nodes[idx_constrained]

        return all_nodes

    def construct_model_from_name(self, given_name):
        # should be able to construct a model purely from a name given as a string
        pass

    def submodel_without_node_idx(self, n):

        if n < 1 :
            print("\t Can't remove head node of a model ")
            return -1
        if len(self._constraints) > 0 :
            print("\t Reduced model of a constrained model is not yet implemented")
            raise NotImplementedError

        new_model = self.copy()
        node_to_remove = (new_model.get_nodes_with_freedom())[n]
        reduced_model = new_model.remove_node(node=node_to_remove)
        return reduced_model

    def remove_node(self, node):

        parent_of_removed = node.parent
        parent_of_removed.children_list.remove(node)
        parent_of_removed.build_name()

        self.build_name()

        return self

    def scipy_func(self, x, *args):
        self.set_args(*args)
        return self.eval_at(x)

    """
    Holonomic Constraints (reduces dof by one)
    """
    @staticmethod
    def unity_constraint(x : float):
        return 1
    @staticmethod
    def sameness_constraint(x: float):
        return x
    # Normalization of A exp[ B(x+C)^2 ] requires A=sqrt(1 / 2 pi sigma^2) with B= - 1 / 2 sigma^2
    @staticmethod
    def gaussian_normalization_constraint1(x: float):
        return np.sqrt(np.abs(x) / np.pi)
    # Alternative normalization of A exp[ B(x+C)^2 ] requires B = -pi*A^2 with A = 1/sqrt(2 pi sigma^2)
    @staticmethod
    def gaussian_normalization_constraint(x: float):
        return -np.pi * np.power(x, 2)

    """
    Non-holonomic constraints (doesn't reduce dof)
    """
    @staticmethod
    def positive_constraint(x: float):
        return x if x > 0 else 1e5
    @staticmethod
    def negative_constraint(x: float):
        return x if x < 0 else -1e5

    @staticmethod
    def built_in_dict():

        built_ins = {}

        # linear model: m, b as parameters (roughly)
        linear = CompositeFunction(func=PrimitiveFunction.built_in("pow1"),
                                   children_list=[PrimitiveFunction.built_in("pow1"),PrimitiveFunction.built_in("pow0")],
                                   name = "Linear Model")

        # Gaussian: A, mu, sigma as parameters (roughly)
        gaussian_inner_negativequadratic = CompositeFunction(func=PrimitiveFunction.built_in("pow2_force_neg_arg"),
                                                             children_list=[PrimitiveFunction.built_in("pow1"),
                                                                            PrimitiveFunction.built_in("pow0")]
                                                            )
        gaussian = CompositeFunction(func=PrimitiveFunction.built_in("exp"),
                                     children_list=[gaussian_inner_negativequadratic],
                                     name = "Gaussian")

        # Normal: mu, sigma as parameters (roughly)
        normal = gaussian.copy()
        normal.name = "Normal"
        normal.add_constraint( (1,CompositeFunction.gaussian_normalization_constraint,0) )

        # Sigmoid H/(1 + exp(-w(x-x0)) ) + F
        # aka F[ 1 + h/( 1+Bexp(-wx) ) ]
        exp_part = CompositeFunction(func=PrimitiveFunction.built_in("exp"),
                                     children_list=[PrimitiveFunction.built_in("pow1")])
        inv_part = CompositeFunction(func=PrimitiveFunction.built_in("pow_neg1_force_pos_arg"),
                                     children_list=[PrimitiveFunction.built_in("pow0"),exp_part])
        sigmoid = CompositeFunction(func=PrimitiveFunction.built_in("pow1"),
                                    children_list=[PrimitiveFunction.built_in("pow0"),inv_part],
                                    name="Sigmoid")

        # make dict entries
        built_ins["Linear"] = linear
        built_ins["Gaussian"] = gaussian
        built_ins["Normal"] = normal
        built_ins["Sigmoid"] = sigmoid

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

    @property
    def dimension_arg(self):
        if self._parent is None:
            return -self.dimension_func
        # else
        if self._parent.func.name[:3] == "pow" :
            if self == self._parent.children_list[0] :
                return 0
            # else
            return self._parent.children_list[0].dimension_func - self.dimension_func
        # else, e.g. self._parent.func.name in ["my_cos","my_sin","my_exp","my_log"] :
        return -self.dimension_func

    @property
    def dimension_func(self):
        if self._children_list == [] :
            if self._prim_of_this_node.name[:3] == "my_" :
                return 0
            elif self._prim_of_this_node.name == "pow0" :
                return 0
            elif self._prim_of_this_node.name == "pow1" :
                return 1
            elif self._prim_of_this_node.name[:4] == "pow2" :
                return 2
            elif self._prim_of_this_node.name == "pow3" :
                return 3
            elif self._prim_of_this_node.name == "pow4" :
                return 4
            elif self._prim_of_this_node.name[:8] == "pow_neg1" :
                return -1
        else :
            if self._prim_of_this_node.name[:3] == "my_":
                return 0
            if self._prim_of_this_node.name == "pow0" :
                return 0
            elif self._prim_of_this_node.name == "pow1" :
                return self._children_list[0].dimension
            elif self._prim_of_this_node.name[:4] == "pow2" :
                return 2*self._children_list[0].dimension
            elif self._prim_of_this_node.name == "pow3" :
                return 3*self._children_list[0].dimension
            elif self._prim_of_this_node.name == "pow4" :
                return 4*self._children_list[0].dimension
            elif self._prim_of_this_node.name[:8] == "pow_neg1" :
                return -1*self._children_list[0].dimension

    @ property
    def dimension(self):
        return self.dimension_arg + self.dimension_func




def test_composite_functions():

    cos_sqr =  CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                  children_list=[PrimitiveFunction.built_in("cos")],
                                  name="cos_sqr")
    inv_lin_cos_sqr = CompositeFunction(func=PrimitiveFunction.built_in("pow_neg1"),
                                        children_list=[PrimitiveFunction.built_in("pow2"),
                                                       PrimitiveFunction.built_in("pow1")],
                                        name="inv_lin_cos_sqr")
    inv_lin_cos_sqr.print_tree()
    print(inv_lin_cos_sqr.tree_as_string_with_dimensions())

    test_comp = CompositeFunction(func=PrimitiveFunction.built_in("pow1"),
                                  children_list=[PrimitiveFunction.built_in("pow0"),
                                                 PrimitiveFunction.built_in("pow1")],
                                  name="TestMeNow")
    test_comp.set_args(5,7)

    print( "pp",test_comp.get_args() )
    test_comp.print_tree()
    value = test_comp.eval_at(1)
    print(value)
    print(test_comp.dimension_func, test_comp.dimension_arg )

    test_comp2 = CompositeFunction(func=PrimitiveFunction.built_in("pow2"),
                                    children_list=[test_comp,test_comp,test_comp],
                                    name="TestMeNowBaby")
    test_comp2.print_tree()
    value2 = test_comp2.eval_at(1)
    print(value2)

    constraint1 = (0,CompositeFunction.sameness_constraint,1)
    test_comp.add_constraint(constraint1)
    test_comp.set_args(5)

    test_comp.print_tree()
    value3 = test_comp.eval_at(5)
    print(value3)



if __name__ == "__main__":
    test_composite_functions()

