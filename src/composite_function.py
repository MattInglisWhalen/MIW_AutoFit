from __future__ import annotations

# built-in libraries
from typing import Callable, Union
import re as regex

# external libraries
import numpy as np
import scipy.special
import scipy.stats

# internal classes
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.package import logger


class CompositeFunction:

    """
    A composite function is represented as a tree.
          f
      |      |
      5     13
    |  |      |
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
    This class represents the nodes of the tree. A node with no children is just a wrapper for a primitive function,
    while a node with no parent is the function tree as a whole.
    """

    _built_in_comps_dict : dict[str,CompositeFunction] = {}

    def __init__(self,
                 children_list : Union[None,list[PrimitiveFunction],list[CompositeFunction]] = None,
                 younger_brother : Union[None,PrimitiveFunction,CompositeFunction] = None,
                 parent : Union[None,CompositeFunction] = None,
                 prim_: PrimitiveFunction = PrimitiveFunction.built_in("sum"),
                 name : str = ""):

        # Declarations before manipulations
        self._children_list : list[CompositeFunction] = []
        self._younger_brother : Union[None,CompositeFunction] = None
        self._older_brother : Union[None,CompositeFunction] = None
        self._parent : Union[None,CompositeFunction] = parent
        self._prim : PrimitiveFunction = prim_.copy()
        self._longname : str = ""
        self._shortname : str = name
        self._constraints : list[tuple[int, Callable[[float], float], int]] = []
        # list of (idx1, func, idx3) triplets, with the interpretation that par[idx1] = func( par[idx2 )]
        # if there are constraints, the add_child and add_brother functions should throw an error

        if children_list :
            for child in children_list :
                self.add_child(child)
        if younger_brother is not None:
            self.add_younger_brother(younger_brother)

        if self.num_children() < 1 and self.younger_brother is None :
            self.build_longname()
        self._shortname = name

        # for being able to reproduce fits of zero-arg functions
        self._is_submodel : bool = False
        self._submodel_zero_index : int = -1
        self._submodel_of : Union[None,CompositeFunction] = None


    def __repr__(self):
        return f"{self._longname} w/ {self.dof} dof{' (submodel)' if self._is_submodel else ''}"

    """
    Properties
    """

    @property
    def children_list(self) -> list[CompositeFunction]:
        return self._children_list
    @property
    def younger_brother(self) -> Union[None,CompositeFunction]:
        return self._younger_brother
    @property
    def older_brother(self) -> Union[None,CompositeFunction]:
        return self._older_brother
    @older_brother.setter
    def older_brother(self,older: CompositeFunction):
        self._older_brother = older
    @property
    def parent(self) -> Union[None,CompositeFunction]:
        return self._parent
    @parent.setter
    def parent(self, par):
        self._parent = par
        if self._younger_brother is not None:
            self._younger_brother.parent = par

    @property
    def prim(self) -> PrimitiveFunction:
        return self._prim
    @prim.setter
    def prim(self, p):
        self._prim = p
        self.build_longname()

    @property
    def args(self):
        return self.get_args()
    @args.setter
    def args(self, other):
        self.set_args(*other)

    @property
    def name(self) -> str:
        if self._shortname == "" :
            return self._longname
        return self._shortname
    @name.setter
    def name(self, val):
        self._shortname = val
    @property
    def shortname(self):
        return self._shortname
    @shortname.setter
    def shortname(self, other):
        self._shortname = other
    @property
    def longname(self):
        return self._longname

    @property
    def constraints(self):
        return self._constraints

    @property
    def dof(self) -> int:
        return self.calculate_degrees_of_freedom()

    def set_submodel_of_zero_idx(self, larger_model, zero_idx):
        if larger_model is None :
            return
        self._is_submodel = True
        self._submodel_zero_index = zero_idx
        self._submodel_of = larger_model.copy()
        if self.name == larger_model.name :
            logger(f"Set submodel: {self.name} =!= {larger_model.name}")
            raise RuntimeError
    @property
    def is_submodel(self) -> bool:
        return self._is_submodel
    @property
    def submodel_zero_index(self) -> int:
        if self._is_submodel :
            assert self._submodel_zero_index is not None
            assert self._submodel_of is not None
        else :
            assert self._submodel_zero_index is None
            assert self._submodel_of is None
        return self._submodel_zero_index
    @property
    def submodel_of(self) -> Union[None,CompositeFunction]:
        return self._submodel_of
    def print_sub_facts(self):
        logger(f"{self}{'is' if self._is_submodel else 'isnt'} "
               f"submodel of {self._submodel_of} {self._submodel_zero_index}")



    """
    Manipulation of the tree
    """

    def add_child(self, child : Union[PrimitiveFunction,CompositeFunction], update_name : bool = True):

        if len(self._constraints) > 0 :
            raise AttributeError

        new_child = child.copy()
        if isinstance(new_child, PrimitiveFunction):  # have to convert it to a Composite
            prim_as_comp = CompositeFunction(prim_=new_child, parent=self)
            self._children_list.append(prim_as_comp)
        else :
            new_child._parent = self
            self._children_list.append(new_child)

        if update_name:
            self.build_longname()

    def add_younger_brother(self, brother_to_add : Union[PrimitiveFunction,CompositeFunction], update_name = True):
        if len(self._constraints) > 0 :
            raise AttributeError

        new_brother = brother_to_add.copy()
        if isinstance(new_brother, PrimitiveFunction):  # have to convert it to a Composite
            new_brother_comp = CompositeFunction(prim_=new_brother)
        else :
            new_brother_comp = new_brother
        if new_brother_comp.prim.name == "sum_" :
            new_brother_comp.prim = PrimitiveFunction.built_in("pow1").copy()

        new_brother_comp.parent = self.parent
        new_brother_comp.older_brother = self
        if self._younger_brother is None:
            self._younger_brother = new_brother_comp
        else:
            self._younger_brother.add_younger_brother(new_brother_comp)

        if update_name:
            self.build_longname()
    def copy(self) -> CompositeFunction:

        new_comp = CompositeFunction(name=self._shortname, prim_=self.prim)
        for child in self._children_list:
            new_comp.add_child(child)
        if self._younger_brother is not None:
            new_comp.add_younger_brother(self.younger_brother)
        # don't copy older siblings
        # don't copy parent
        for constraint in self._constraints:
            new_comp.add_constraint(constraint)

        new_comp.set_submodel_of_zero_idx(self._submodel_of, self._submodel_zero_index)

        return new_comp

    def build_longname(self):

        name_str = f"{self.prim.name}("

        names = sorted([child.longname for child in self._children_list],reverse=True)
        for name in names:
            name_str += f"{name}+"
        if len(self._children_list) > 0:
            name_str = name_str[:-1] + "))"
        name_str = name_str[:-1]

        if self._younger_brother is not None :
            younger_name = self._younger_brother.longname
            name_str = '·'.join(sorted([name_str,younger_name],reverse=True))

        if (self.parent is None and self.num_children() > 0
                and self.prim.name == "sum_" and self.younger_brother is None) :
            name_str = name_str[5:-1]

        self._longname = name_str

        if self.parent is not None:
            self.parent.build_longname()
        if self.older_brother is not None :
            self._older_brother.build_longname()

    def add_constraint(self, constraint_3tuple: tuple[int, Callable[[float], float], int] ):
        self._constraints.append(constraint_3tuple)

    """
    Info about the composite
    """

    def num_nodes(self) -> int:
        nodes = 1  # for self
        for child in self._children_list:
            nodes += child.num_nodes()
        if self._younger_brother is not None :
            nodes += self._younger_brother.num_nodes()

        return nodes
    def get_node_with_index(self, n) -> CompositeFunction:
        if n == 0 :
            return self
        it = 1
        for child in self._children_list :
            if it <= n < it + child.num_nodes() :
                return child.get_node_with_index(n-it)
            it += child.num_nodes()
        if self._younger_brother is not None :
            return self._younger_brother.get_node_with_index(n-it)
        raise IndexError
    def get_nodes_with_freedom(self, skip_flag=0):
        # untested with siblings
        # gets all nodes which are associated with a degree of freedom
        all_nodes = []
        if skip_flag or self._prim.name == "sum_":
            pass
        else :
            all_nodes.append(self)

        skip_flag = 0
        if self._prim.name[0:3] == "pow" and len(self._children_list) > 0 :
            skip_flag = 1

        for child in self._children_list :
            all_nodes.extend( child.get_nodes_with_freedom(skip_flag) )
            skip_flag = 0

        if self._younger_brother is not None:
            all_nodes.extend( self._younger_brother.get_nodes_with_freedom(skip_flag=1) )

        for idx_constrained, _, _ in sorted(self._constraints, key=lambda tup: tup[0], reverse=True) :
            del all_nodes[idx_constrained]

        return all_nodes

    def calculate_degrees_of_freedom(self) -> int:

        num_dof = 1 if self.older_brother is None else 0
        for child in self._children_list:
            num_dof += child.calculate_degrees_of_freedom()
        if self._younger_brother is not None :
            num_dof += self._younger_brother.calculate_degrees_of_freedom()

        if self._prim.name[:4] == "sum_" :
            # sums are special: they have no degrees of freedom by themselves
            num_dof -= 1
        if self._prim.name[:3] == "pow" and len(self._children_list) > 0:
            # powers are special: they lose a degree of freedom if they have children
            # e.g. A*( B*cos(x) )^2 == A*B^2*cos^2(x) == C*cos^2(x). We keep the outer one as free,
            # and the first parameter in the composition is set to unity
            num_dof -= 1

        return num_dof - len(self._constraints)

    def tree_as_string(self, buffer_chars=""):

        buff_num = 10
        info_str = f"{self._prim.name[:10] : <10}"  # pads and truncates to ensure a length of 10

        bools = self.parent is not None, self.older_brother is not None, self.younger_brother is not None

        if bools == (True, True, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (True, True, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (True, False, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" ~ {info_str}"
        elif bools == (True, False, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" ~ {info_str}"

        elif bools == (False, True, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (False, True, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (False, False, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f"   {info_str}"
        elif bools == (False, False, False) :
            next_buffer = "" + " " * buff_num
            tree_str = f"{info_str}"
        else :
            next_buffer = ""
            tree_str = ""

        for idx, child in enumerate(self._children_list):
            if idx > 0:
                tree_str += buffer_chars+next_buffer
            tree_str += child.tree_as_string(buffer_chars=buffer_chars+next_buffer)

        if self.younger_brother is not None :
            tree_str += '\n' + buffer_chars
            bro_str = self._younger_brother.tree_as_string(buffer_chars=buffer_chars)
            tree_str += bro_str

        # return tree_str.rstrip('\n') + "\n"
        return regex.sub(f"\n\n\n*",'\n',tree_str).rstrip('\n') + '\n'
    def tree_as_string_with_args(self, buffer_chars=""):

        buff_num = 19
        info_str = f"{self._prim.arg:+.2E}{self._prim.name[:10] : <10}"  # pads and truncates to ensure a length of 10

        bools = self.parent is not None, self.older_brother is not None, self.younger_brother is not None

        if bools == (True, True, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (True, True, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (True, False, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" ~ {info_str}"
        elif bools == (True, False, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" ~ {info_str}"

        elif bools == (False, True, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (False, True, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (False, False, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f"   {info_str}"
        elif bools == (False, False, False) :
            next_buffer = "" + " " * buff_num
            tree_str = f"{info_str}"
        else :
            next_buffer = ""
            tree_str = ""

        for idx, child in enumerate(self._children_list):
            if idx > 0:
                tree_str += buffer_chars+next_buffer
            tree_str += child.tree_as_string_with_args(buffer_chars=buffer_chars+next_buffer)

        if self.younger_brother is not None :
            tree_str += '\n' + buffer_chars
            bro_str = self._younger_brother.tree_as_string_with_args(buffer_chars=buffer_chars)
            tree_str += bro_str

        # return tree_str.rstrip('\n') + "\n"
        return regex.sub(f"\n\n\n*",'\n',tree_str).rstrip('\n') + '\n'
    def tree_as_string_with_dimensions(self, buffer_chars=""):

        buff_num = 15
        info_str = f"{self.dimension_arg:+}/{self.dimension_func:+}{self._prim.name[:10] : <10}"

        bools = self.parent is not None, self.older_brother is not None, self.younger_brother is not None

        if bools == (True, True, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (True, True, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (True, False, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" ~ {info_str}"
        elif bools == (True, False, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" ~ {info_str}"

        elif bools == (False, True, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (False, True, False) :
            next_buffer = "   " + " " * buff_num
            tree_str = f" x {info_str}"
        elif bools == (False, False, True) :
            next_buffer = " | " + " " * buff_num
            tree_str = f"   {info_str}"
        elif bools == (False, False, False) :
            next_buffer = "" + " " * buff_num
            tree_str = f"{info_str}"
        else :
            next_buffer = ""
            tree_str = ""

        for idx, child in enumerate(self._children_list):
            if idx > 0:
                tree_str += buffer_chars+next_buffer
            tree_str += child.tree_as_string_with_dimensions(buffer_chars=buffer_chars+next_buffer)

        if self.younger_brother is not None :
            tree_str += '\n' + buffer_chars
            bro_str = self._younger_brother.tree_as_string_with_dimensions(buffer_chars=buffer_chars)
            print(f"~~{bro_str}~~")
            tree_str += bro_str

        # return tree_str.rstrip('\n') + "\n"
        return regex.sub(f"\n\n\n*",'\n',tree_str).rstrip('\n') + '\n'
    def print_tree(self):
        logger(f"{self.name}:")
        # logger(self.tree_as_string())
        logger(self.tree_as_string_with_args())


    """
    Metadata for tree trimming
    """

    @property
    def depth(self):
        if self.num_children() < 1 :
            return 0
        curr_max = 0
        for child in self._children_list :
            child_depth = max( [bro.depth for bro in child.self_and_all_youngers() ] )
            if curr_max < child_depth :
                curr_max = child_depth
        return 1 + curr_max

    @property
    def width(self):
        if self.num_children() == 0 :
            if self.younger_brother is not None :
                return 1+self.younger_brother.width
            return 1
        if self.younger_brother is None:
            return sum([child.width for child in self._children_list])
        return self.younger_brother.width + sum([child.width for child in self._children_list])

    def has_trig_children(self):
        for child in self._children_list :
            if "sin" in child.name or "cos" in child.name :
                return True
        return False
    def has_double_trigness(self):
        if "sin" in self.prim.name or "cos" in self.prim.name :
            if self.has_trig_children() :
                return True
        for child in self._children_list :
            if child.has_double_trigness():
                return True
        if self._younger_brother is not None:
            return self._younger_brother.has_double_trigness()
        return False
    def num_trig(self):
        return self.num_cos() + self.num_sin()
    def num_cos(self):
        return self.longname.count("cos")
    def num_sin(self):
        return self.longname.count("sin")

    def has_exp_children(self):
        for child in self._children_list :
            if "exp" in child.name :
                return True
        return False
    def has_double_expness(self):
        if "exp" in self.prim.name :
            if self.has_exp_children() :
                return True
        for child in self._children_list :
            if child.has_double_expness():
                return True
        if self._younger_brother is not None:
            return self._younger_brother.has_double_expness()
        return False

    def has_log_children(self):
        for child in self._children_list :
            if "log" in child.name :
                return True
        return False
    def has_double_logness(self):
        if "log" in self.prim.name :
            if self.has_log_children() :
                return True
        for child in self._children_list :
            if child.has_double_logness():
                return True
        if self._younger_brother is not None:
            return self._younger_brother.has_double_logness()
        return False

    def self_and_all_youngers(self):
        brothers_list = [self]
        if self.younger_brother is not None:
            brothers_list.extend( self.younger_brother.self_and_all_youngers() )
        return brothers_list

    def has_argless_explike(self):
        if self.prim.name in ["my_cos","my_sin","my_exp","my_log"] and self.num_children() < 1 :
            if self.prim.name == "my_log" and self.parent is not None :
                if self.parent.prim.name == "my_exp" :
                    return False
                # elif (self.parent.prim.name[:3] == "pow"
                #         and self.parent.parent is not None
                #         and self.parent.parent.prim.name == "my_exp"):
                else :
                    return True
            return True
        if any([child.has_argless_explike() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_argless_explike()
        return False
    def has_trivial_pow1(self):
        if self.prim.name == "pow1" and (self.num_children() == 1
                                         or (self.num_children() > 1 and self.younger_brother is None)) :
            return True
        if any([child.has_trivial_pow1() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_trivial_pow1()
        return False
    def has_repeated_reciprocal(self):
        if self.prim.name == "pow_neg1" and self.num_children() == 1 :
            child = self.children_list[0]
            if any( bro.prim.name in ["pow_neg1","my_exp"] for bro in child.self_and_all_youngers() ) :
                return True
        if any([child.has_repeated_reciprocal() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_repeated_reciprocal()
        return False
    def has_reciprocal_cancel_pospow(self):
        if (self.prim.name == "pow_neg1" and self.younger_brother is not None
                and self.younger_brother.prim.name in ["pow1","pow2","pow3","pow4"]
                and self.younger_brother.num_children() < 1):
            return True
        if any([child.has_reciprocal_cancel_pospow() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_reciprocal_cancel_pospow()
        return False
    def has_log_with_odd_power(self):
        if (self.prim.name == "my_log" and self.num_children() == 1
                and self.children_list[0].prim.name in ["pow_neg1","pow3"] ):
            child = self.children_list[0]
            if (child.younger_brother is None or
                    (child.younger_brother.prim.name in ["pow_neg1","pow3"]  # log(pow_neg1 . pow_neg1) = log(pow1^2)
                     and child.younger_brother.younger_brother is None) ):
                return True
        if any([child.has_log_with_odd_power() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_log_with_odd_power()
        return False
    def has_pow_with_no_sum(self):  # exludes cos, sin, log
        if (self.prim.name[:3] == "pow" and self.num_children() == 1
                and self.children_list[0].prim.name[:4] in ["my_e","pow0","pow1","pow_","pow2","pow3","pow4"]
                and self.children_list[0].younger_brother is None):
                return True
        if any([child.has_pow_with_no_sum() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_pow_with_no_sum()
        return False
    def has_exp_with_pow0(self):
        if self.prim.name == "my_exp" and "pow0" in [icomp.prim.name for icomp in self._children_list]:
                return True
        if any([child.has_exp_with_pow0() for child in self._children_list]) :
            return True
        if self.younger_brother is not None :
            return self.younger_brother.has_exp_with_pow0()
        return False

    def num_children(self):
        return len(self._children_list)



    """
    Evaluation
    """

    def eval_at(self,x, X0 = 0, Y0 = 0):
        if X0 :
            # logger(f"{X0=}")
            # the model is working with LX as the independent variable, but we're being passed x
            LX = np.log(x/X0)
            x = LX
        children_eval_to = 0
        if len(self._children_list) == 0 :
            if Y0 :
                # the model is working with LY as the dependent variable, but we need to return y
                LY = self._prim.eval_at(x)
                if self._younger_brother is not None :
                    LY *= self._younger_brother.eval_at(x)
                y = Y0*np.exp(LY)
            else:
                y = self._prim.eval_at(x)
                if self._younger_brother is not None :
                    y *= self._younger_brother.eval_at(x)
            return y
        for child in self._children_list :
            children_eval_to += child.eval_at(x)
        if Y0 :
            # logger(f"{Y0=}")
            # the model is working with LY as the dependent variable, but we're expecting to return x
            LY = self._prim.eval_at(children_eval_to)
            if self._younger_brother is not None:
                LY *= self._younger_brother.eval_at(x)
            y = Y0*np.exp(LY)
        else:
            y = self._prim.eval_at(children_eval_to)
            if self._younger_brother is not None:
                y *= self._younger_brother.eval_at(x)
        return y
    def eval_deriv_at(self,x):
        # simple symmetric difference. If exact derivative is needed, can add that later
        delta = 1e-5
        return (self.eval_at(x + delta) - self.eval_at(x - delta)) / (2 * delta)


    def set_args(self, *args):

        it = 0
        args_as_list = list(args)

        # insert zeroes where the constrained arguments go
        for idx_constrained, _, _ in sorted(self._constraints, key=lambda tup: tup[0]) :
            args_as_list.insert( idx_constrained, 0 )
        # apply the constraints
        for idx_constrained, func, idx_other in sorted(self._constraints, key=lambda tup: tup[0]) :
            args_as_list[idx_constrained] = func( args_as_list[idx_other] )

        if self.older_brother is None and self._prim.name != "sum_" :
            self._prim.arg = args_as_list[it]
            it += 1
        else :
            self._prim.arg = 1

        if self._prim.name[0:3] == "pow" and len(self._children_list) > 0 :
            args_as_list.insert(it,1)

        for child in self._children_list :
            next_dof = child.dof
            child.set_args( *args_as_list[it:it+next_dof] )
            it += next_dof

        if self._younger_brother is not None :
            brother_dof = self._younger_brother.dof
            self._younger_brother.set_args( *args_as_list[it:it+brother_dof] )
            it += brother_dof

        if it != len(args_as_list) :
            logger(f"Trying to set {args_as_list=} in {self.name}")
            raise RuntimeError
    def get_args(self, skip_flag=0) -> list[float]:
        # get all arguments normally, then pop off the ones with constraints once we get to the head
        all_args = []
        if skip_flag or self._prim.name == "sum_":
            pass
        else :
            all_args.append(self._prim.arg)

        skip_flag = 0
        if self._prim.name[0:3] == "pow" and len(self._children_list) > 0 :
            skip_flag = 1

        for child in self._children_list :
            all_args.extend( child.get_args(skip_flag) )
            skip_flag = 0

        if self._younger_brother is not None:
            all_args.extend( self._younger_brother.get_args(skip_flag=1) )

        for idx_constrained, _, _ in sorted(self._constraints, key=lambda tup: tup[0], reverse=True) :
            del all_args[idx_constrained]

        return all_args

    @staticmethod
    def construct_model_from_str(form: str,
                                 error_handler: Callable[[str],bool],
                                 name: str = "") -> Union[None, CompositeFunction]:

        logger(f"Entering construction for {form}")

        # get all the names of the primitives used
        split_form = regex.split(f"[·+()*]", form)
        prim_names = [x for x in split_form if x]

        # check that the name can be used as a function
        prim_list = []
        error_handler(" ")
        for prim in prim_names:
            valid = False
            library_list = [np,scipy.special,scipy.stats._continuous_distns]
            library_names = ["numpy","scipy.special","scipy.stats"]  # math doesn't accept arrays of xvals as inputs

            if not valid and prim in PrimitiveFunction.built_in_dict():
                # logger(f"{prim} exists in PrimitiveFunction")
                try :
                    fn = PrimitiveFunction.built_in(prim).copy()
                    fn.eval_at(np.pi/4)
                except TypeError :
                    error_handler(f"<autofit> function {prim} has multiple arguments, "
                                  f"or only takes integers as input.")
                else :
                    prim_list.append(fn)
                    valid = True
            for lib, lib_name in zip(library_list, library_names) :
                if valid :
                    break
                if getattr(lib, prim, None) is not None:
                    # logger(f"{prim} exists in {lib_name}")
                    try :
                        if lib == scipy.stats._continuous_distns :
                            getattr(lib, prim).pdf(np.pi / 4)
                        else :
                            getattr(lib,prim)(np.pi/4)
                    except TypeError :
                        error_handler(f"<{lib_name}> function {prim} has multiple arguments, "
                                      f"or only takes integers as input.")
                    else :
                        if lib == scipy.stats._continuous_distns:
                            prim_list.append(PrimitiveFunction(other_callable=getattr(lib, prim).pdf, name=prim))
                        else :
                            prim_list.append( PrimitiveFunction(other_callable=getattr(lib, prim),name=prim) )
                        valid = True

            if not valid :
                logger(PrimitiveFunction.built_in_dict())
                error_handler(f"Couldn't find a valid version of \"{prim}\" in the list of known functions.")
                # error_handler(f"  You can try creating it yourself using the Custom Function button.")
                # noinspection PyTypeChecker
                return None

        form_it, prim_it = 0, 0
        last_char = "+"
        man_model = CompositeFunction(name=name,prim_=PrimitiveFunction.built_in("sum"))

        while form_it < len(form) :

            c = form[form_it]
            if c == '(' :
                sub_form = ""
                open_paren = 1
                while open_paren > 0 :
                    c = form[form_it+1]
                    if c == '(':
                        open_paren += 1
                    elif c == ')':
                        open_paren -= 1
                    sub_form += c
                    form_it += 1

                sub_model = CompositeFunction.construct_model_from_str(form=sub_form[:-1], error_handler=error_handler)
                if sub_model is None:
                    error_handler(f"{form} had a problem creating sub-model with {sub_form[:-1]}")
                    return None

                if last_char == '+' :
                    man_model.add_child(sub_model)
                elif last_char in ['·','*'] :
                    man_model.children_list[-1].add_younger_brother(sub_model)
                else :
                    if sub_model.younger_brother is None :
                        for sub_child in sub_model.children_list :
                            man_model.children_list[-1].add_child(sub_child)
                    else :
                        man_model.children_list[-1].add_child(sub_model)
                last_char = c
                form_it += 1
            elif c in ['+','·','*'] :
                last_char = c
                form_it += 1
            else :
                prim_to_add_name = c
                while form[form_it+1] not in ['+','·','*','('] :
                    c = form[form_it+1]
                    prim_to_add_name += c
                    form_it += 1
                    if form_it+1 == len(form) :
                        logger(prim_list)
                        break

                if form_it+1 == len(form) or form[form_it+1] in ['+','·','*'] :
                    for prim in prim_list[:] :
                        if prim_to_add_name in prim.name :
                            prim_to_add = prim
                            prim_list.remove(prim)
                            break
                    else :
                        error_handler(f"Couldn't find primitive for {prim_to_add_name} in {prim_list}")
                        return None
                    if last_char == '+':
                        man_model.add_child(prim_to_add)
                    elif last_char in ['·', '*']:
                        man_model.children_list[-1].add_younger_brother(prim_to_add)
                    else:
                        man_model.children_list[-1].add_child(prim_to_add)
                    last_char = c
                    form_it += 1
                elif form[form_it+1] == '(' :
                    # it's actually a composite, we continue until we reach the closing brace

                    composed_fn_name = "("
                    form_it += 1
                    open_paren2 = 1
                    while open_paren2 > 0 :
                        c = form[form_it + 1]
                        if c == '(' :
                            open_paren2 += 1
                        elif c == ')' :
                            open_paren2 -= 1
                        composed_fn_name += c
                        form_it += 1
                        if form_it + 1 == len(form):
                            break
                    prim_with_composition = CompositeFunction.construct_model_from_str(prim_to_add_name,
                                                                                       error_handler=error_handler)
                    composed_fn = CompositeFunction.construct_model_from_str(composed_fn_name,
                                                                             error_handler=error_handler)
                    if prim_with_composition is None or composed_fn is None :
                        error_handler(f"{form} had a problem creating sub-function "
                                      f"with {prim_to_add_name} or {composed_fn_name}")
                        return None
                    if composed_fn.prim.name == "sum_" and composed_fn.younger_brother is None :
                        for composed_child in composed_fn.children_list :
                            prim_with_composition.add_child(composed_child)
                    else :
                        prim_with_composition.add_child(composed_fn)
                    if last_char == '+':
                        man_model.add_child(prim_with_composition)
                    elif last_char in ['·', '*']:
                        man_model.children_list[-1].add_younger_brother(prim_with_composition)
                    else:
                        man_model.children_list[-1].add_child(prim_with_composition)
                    last_char = ')'
                    form_it += 1
                else :
                    raise RuntimeError

        # man_model.print_tree()

        if man_model.prim.name == "sum_" and man_model.num_children() == 1 and man_model.younger_brother is None :
            man_model = man_model.children_list[0]
            man_model.parent = None

        error_handler("")
        return man_model

    def submodel_without_node_idx(self, n) -> Union[None, CompositeFunction]:

        if len(self._constraints) > 0 :
            logger("Reduced model of a constrained model is not yet implemented")
            raise NotImplementedError

        new_model = self.copy()
        if new_model.shortname != "" :
            new_model.shortname = "Trimmed{" + new_model.name + "}"
        else:
            new_model.build_longname()
        node_to_remove = (new_model.get_nodes_with_freedom())[n]

        # untested with siblings
        if node_to_remove.parent is None and node_to_remove.older_brother is None :
            logger("Can't remove head node of a model ")
            return None

        reduced_model = new_model.remove_node(node=node_to_remove)
        return reduced_model

    def remove_node(self, node: CompositeFunction) -> CompositeFunction:

        # untested with siblings
        parent_of_removed = node.parent
        parent_of_removed.children_list.remove(node)

        top = parent_of_removed
        while top.parent is not None :
            top.shortname = ""
            top = top.parent

        parent_of_removed.build_longname()

        self.build_longname()

        return self

    def scipy_func(self, x, *args):
        self.set_args(*args)
        return self.eval_at(x)

    @staticmethod
    def build_built_in_dict() -> None:

        # linear model: m, b as parameters (roughly)
        linear = CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                          children_list=[PrimitiveFunction.built_in("pow1"),
                                                         PrimitiveFunction.built_in("pow0")],
                                          name="Linear")

        # Gaussian: A, mu, sigma as parameters (roughly)
        prim_dim0_pow2_neg = PrimitiveFunction(func=PrimitiveFunction.dim0_pow2)  # sigma
        prim_pow1_shift = PrimitiveFunction(func=PrimitiveFunction.pow1_shift)    # mu
        gaussian_inner_negativequadratic = CompositeFunction(prim_=prim_dim0_pow2_neg,
                                                             children_list=[prim_pow1_shift])
        gaussian = CompositeFunction(prim_=PrimitiveFunction.built_in("exp"),     # A
                                     children_list=[gaussian_inner_negativequadratic],
                                     name="Gaussian")

        # Normal: mu, sigma as parameters (roughly)
        prim_outer_gaussian = PrimitiveFunction(func=PrimitiveFunction.n_exp_dim2)  # sigma
        normal = CompositeFunction(prim_=prim_outer_gaussian,
                                   children_list=[prim_pow1_shift],                 # mu
                                   name="Normal")

        # Sigmoid H/(1 + exp(-w(x-x0)) ) + F
        # aka F[ 1 + h/( 1+Bexp(-wx) ) ]
        prim_exp_dim1 = PrimitiveFunction(func=PrimitiveFunction.exp_dim1)            # w
        exp_part = CompositeFunction(prim_=prim_exp_dim1,
                                     children_list=[prim_pow1_shift])                 # x0
        inv_part = CompositeFunction(prim_=PrimitiveFunction.built_in("pow_neg1"),    # H
                                     children_list=[PrimitiveFunction.built_in("pow0"),exp_part])
        sigmoid = CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),          # F
                                    children_list=[PrimitiveFunction.built_in("pow0"),inv_part],
                                    name="Sigmoid")

        # make dict entries
        CompositeFunction._built_in_comps_dict["Linear"] = linear
        CompositeFunction._built_in_comps_dict["Gaussian"] = gaussian
        CompositeFunction._built_in_comps_dict["Normal"] = normal
        CompositeFunction._built_in_comps_dict["Sigmoid"] = sigmoid
    @staticmethod
    def built_in_list() -> list[CompositeFunction]:
        built_ins = []
        if not CompositeFunction._built_in_comps_dict :
            CompositeFunction.build_built_in_dict()
        for key, comp in CompositeFunction.built_in_dict().items():
            built_ins.append(comp)
        return built_ins
    @staticmethod
    def built_in_dict() -> dict[str,CompositeFunction]:
        return CompositeFunction._built_in_comps_dict
    @staticmethod
    def built_in(key) -> CompositeFunction:
        if not CompositeFunction._built_in_comps_dict :
            CompositeFunction.build_built_in_dict()

        if key in ["Gaussian", "Gaussian1"] :
            return CompositeFunction._built_in_comps_dict["Gaussian"]
        elif key[:8] == "Gaussian" :
            modal = int(key[8:])
            summed_gaussians = []
            for _ in range(modal) :
                summed_gaussians.append( CompositeFunction._built_in_comps_dict["Gaussian"] )
            if modal == 2 :
                prestr = "Bim"
            elif modal == 3 :
                prestr = "Trim"
            else :
                prestr = f"{modal}-M"
            return CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                     children_list=summed_gaussians,
                                     name=f"{prestr}odal-Gaussian")

        if key[:10] == "Polynomial" :
            degree = int(key[10:])
            if degree == 0 :
                return CompositeFunction(prim_=PrimitiveFunction.built_in("pow0"),
                                         name="Polynomial0")


            new_kids_list = []
            for d in range(degree+1) :
                new_kid = PrimitiveFunction.built_in(f"Pow{degree-d}")
                new_kids_list.append(new_kid)

            return CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                     children_list=new_kids_list,
                                     name=f"Polynomial{degree}")

        return CompositeFunction.built_in_dict()[key]

    @property
    def net_function_dimension_self_and_younger_siblings(self) -> int:
        dim = self.dimension_func
        if self._younger_brother is not None :
            dim += self._younger_brother.net_function_dimension_self_and_younger_siblings
        return dim
    @property
    def dimension_arg(self) -> int:

        # special cases
        if self._prim.name in ["dim0_pow2", "pow1_shift", "exp_dim1" ,"n_exp_dim2"] :
            return 1

        if self._older_brother is not None :
            return 0
        if self.parent is None:
            return -self.net_function_dimension_self_and_younger_siblings
        # else
        if self.parent.prim.name[:3] == "pow" :
            if self == self.parent.children_list[0] :
                return 0
            # else
            return (self.parent.children_list[0].net_function_dimension_self_and_younger_siblings
                    - self.net_function_dimension_self_and_younger_siblings)
        # else, e.g. self.parent.func.name in ["my_cos","my_sin","my_exp","my_log"] :
        return -self.net_function_dimension_self_and_younger_siblings
    @property
    def dimension_func(self) -> int:
        # untested with siblings
        if self._prim.name[:3] == "my_" or "sin" in self._prim.name or "cos" in self._prim.name :
            return 0
        elif self._prim.name == "pow0":
            return 0
        elif self._prim.name in ["dim0_pow2", "pow1_shift", "exp_dim1" ,"n_exp_dim2"] :
            return 1

        if not self._children_list:
            if self._prim.name[:4] == "pow1" :
                return 1
            elif self._prim.name[:4] == "pow2" :
                return 2
            elif self._prim.name == "pow3" :
                return 3
            elif self._prim.name == "pow4" :
                return 4
            elif self._prim.name[:8] == "pow_neg1" :
                return -1
            elif self._prim.name[:3] == "Pow" :
                return int(self._prim.name[3:])
            elif self._prim.name == "sum_" :
                return -100  # should never have a sum with no children
        else :
            if self._prim.name[:4] == "pow1" :
                return self._children_list[0].dimension
            elif self._prim.name[:4] == "pow2" :
                return 2*self._children_list[0].dimension
            elif self._prim.name == "pow3" :
                return 3*self._children_list[0].dimension
            elif self._prim.name == "pow4" :
                return 4*self._children_list[0].dimension
            elif self._prim.name[:8] == "pow_neg1" :
                return -1*self._children_list[0].dimension
            elif self._prim.name[:3] == "Pow" :
                return int(self._prim.name[3:])*self._children_list[0].dimension
            elif self._prim.name == "sum_" :
                return self._children_list[0].dimension

        # probably a custom function
        return 0
    @ property
    def dimension(self) -> int:
        return self.dimension_arg + self.net_function_dimension_self_and_younger_siblings


# def sameness_constraint(x):  # to delete
#     return x
# # Normalization of A exp[ B(x+C)^2 ] requires A=sqrt(1 / 2 pi sigma^2) and B= - 1 / 2 sigma^2
# def gaussian_normalization_constraint1(x):  # to delete
#     return np.sqrt(np.abs(x)/np.pi)
# def gaussian_normalization_constraint(x):  # to delete
#     return -np.pi*np.power(x,2)


def do_new_things():

    import random as rng
    sigma = 0.1
    positions = np.arange(-20,20,0.4)
    values = [ rng.normalvariate( mu=5/(1+np.exp(-(x-3)/7))-2.5, sigma=sigma) for x in positions]

    for pos, val in zip(positions,values) :
        logger(f"{pos:.2F}, {val:.2F}, {sigma}")



if __name__ == "__main__" :

    do_new_things()
