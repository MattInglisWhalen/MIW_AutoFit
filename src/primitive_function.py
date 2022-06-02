
# built-in libraries
import random as rng
from dataclasses import field

# external libraries
import numpy as np
from cmath import sin, cos, exp, log


class PrimitiveFunction:

    """
    A wrapper for basic single-variable functions with one parameter

    Instance variables

    _name : string
    _func : a callable function with the form f(x, arg), e.g. A*x^2
    _arg  : the argument for the function
    _deriv : another PrimitiveFunction which is the derivative of self.
                    If left as None, it will be found numerically when required

    Static Variables

    _func_list : a list of all possible

    """

    _built_in_prims_dict = {}

    def __init__(self, name="", func=None, arg=1, deriv=None):
        self._name = name
        if name == "" :
            self._name = func.__name__
        self._func = func  # valid functions must all be of the type f(x,arg)
        self._arg = arg
        self._deriv = deriv

    def __repr__(self):
        return f"Function {self._name} uses {self._func.__name__}(x,arg) with coefficient {self._arg}"


    @property
    def name(self):
        # Try to keep names less than 10 characters, so a composite function's tree looks good
        return self._name

    @property
    def f(self):
        return self._func
    @f.setter
    def f(self, other):
        self._func = other

    @property
    def arg(self):
        return self._arg
    @arg.setter
    def arg(self, val):
        self._arg = val

    @property
    def deriv(self):
        return self._deriv
    @deriv.setter
    def deriv(self, other):
        self._deriv = other

    @property
    def derivative(self):
        prim_deriv = None
        if self._func == PrimitiveFunction.pow0 :
            prim_deriv = PrimitiveFunction( arg=0, func=PrimitiveFunction.pow0)
        elif self._func == PrimitiveFunction.pow1 :
            prim_deriv = PrimitiveFunction( arg=self.arg, func=PrimitiveFunction.pow0)
        elif self._func == PrimitiveFunction.pow2 :
            prim_deriv = PrimitiveFunction( arg=2*self.arg, func=PrimitiveFunction.pow1)
        return prim_deriv

    def eval_at(self,x):
        return self._func(x, self._arg)

    def eval_deriv_at(self,x):
        if self._deriv is not None:
            return self._deriv(x, self._arg)
        else :
            # simple symmetric difference
            delta = 1e-5
            return (self.eval_at(x+delta) - self.eval_at(x-delta) ) / (2*delta)
            # can do higher differences later? https://en.wikipedia.org/wiki/Finite_difference_coefficient
            # return ( self.eval_at(x-2*delta) - 8*self.eval_at(x-delta)
            #           + 8*self.eval_at(x+delta) - self.eval_at(x+2*delta) ) / (12*delta)

    def copy(self):
        new_prim = PrimitiveFunction( name=self._name, func=self._func, arg=self.arg, deriv=self.deriv)
        return new_prim

    """
    Static methods
    """

    @staticmethod
    def pow_neg1(x, arg):
        return arg/x
    @staticmethod
    def pow0(x, arg):
        return arg*x**0
    @staticmethod
    def pow1(x, arg):
        return arg*x
    @staticmethod
    def pow2(x, arg):
        return arg*x*x
    @staticmethod
    def pow3(x, arg):
        return arg*x*x*x
    @staticmethod
    def pow4(x, arg):
        return arg*x*x*x*x
    @staticmethod
    def pow_neg1_force_pos_arg(x, arg):
        return arg/x if arg > 0 else 1e5
    @staticmethod
    def pow2_force_neg_arg(x, arg):
        return arg*x*x if arg < 0 else 1e5
    @staticmethod
    def my_sin(x, arg):
        # arg^2 is needed to break the tie between A*sin(omega*t) and -A*sin(-omega*t)
        return arg*np.sin(x)
    @staticmethod
    def my_cos(x, arg):
        # there is a tie between A*cos(omega*t) and A*cos(-omega*t). How to break this tie?
        return arg*np.cos(x)
    @staticmethod
    def my_exp(x, arg):
        try :
            return arg*np.exp(x)
        except RuntimeWarning :
            print(f"my_exp: {x} is large")
            return 1e10

    @staticmethod
    def my_log(x, arg):
        return arg*np.log(x)

    # arbitrary powers can be created using function composition,
    # i.e use the CompositeFunction class, e.g. exp( 1.5 log(x) ) == x^1.5


    @staticmethod
    def build_built_in_dict():

        # Powers
        prim_pow_neg1 = PrimitiveFunction(func=PrimitiveFunction.pow_neg1 )
        prim_pow_neg1_force_pos_arg = PrimitiveFunction(func=PrimitiveFunction.pow_neg1_force_pos_arg )
        prim_pow0 = PrimitiveFunction(func=PrimitiveFunction.pow0 )
        prim_pow1 = PrimitiveFunction(func=PrimitiveFunction.pow1 )
        prim_pow2 = PrimitiveFunction(func=PrimitiveFunction.pow2 )
        prim_pow2_force_neg_arg = PrimitiveFunction(func=PrimitiveFunction.pow2_force_neg_arg )
        prim_pow3 = PrimitiveFunction(func=PrimitiveFunction.pow3 )
        prim_pow4 = PrimitiveFunction(func=PrimitiveFunction.pow4 )

        PrimitiveFunction._built_in_prims_dict["pow_neg1"] = prim_pow_neg1
        PrimitiveFunction._built_in_prims_dict["pow_neg1_force_pos_arg"] = prim_pow_neg1_force_pos_arg
        PrimitiveFunction._built_in_prims_dict["pow0"] = prim_pow0
        PrimitiveFunction._built_in_prims_dict["pow1"] = prim_pow1
        PrimitiveFunction._built_in_prims_dict["pow2"] = prim_pow2
        PrimitiveFunction._built_in_prims_dict["pow2_force_neg_arg"] = prim_pow2_force_neg_arg
        PrimitiveFunction._built_in_prims_dict["pow3"] = prim_pow3
        PrimitiveFunction._built_in_prims_dict["pow4"] = prim_pow4

        # Trig
        prim_sin = PrimitiveFunction(func=PrimitiveFunction.my_sin )
        prim_cos = PrimitiveFunction(func=PrimitiveFunction.my_cos )

        PrimitiveFunction._built_in_prims_dict["sin"] = prim_sin
        PrimitiveFunction._built_in_prims_dict["cos"] = prim_cos

        # Exponential
        prim_exp = PrimitiveFunction(func=PrimitiveFunction.my_exp )
        prim_log = PrimitiveFunction(func=PrimitiveFunction.my_log )

        PrimitiveFunction._built_in_prims_dict["exp"] = prim_exp
        PrimitiveFunction._built_in_prims_dict["log"] = prim_log

        return PrimitiveFunction._built_in_prims_dict

    @staticmethod
    def built_in_list():
        built_ins = []
        if not PrimitiveFunction._built_in_prims_dict :
            PrimitiveFunction.build_built_in_dict()
        for key, prim in PrimitiveFunction._built_in_prims_dict.items():
            built_ins.append( prim )
        return built_ins

    @staticmethod
    def built_in_dict():
        return PrimitiveFunction._built_in_prims_dict

    @staticmethod
    def built_in(key):
        if not PrimitiveFunction._built_in_prims_dict :
            PrimitiveFunction.build_built_in_dict()
        return PrimitiveFunction._built_in_prims_dict[key]




def test_primitive_functions():

    built_ins = PrimitiveFunction.built_in_list()
    for prim in built_ins:
        print(f"{prim.name}: {prim.eval(0.1)}")


if __name__ == "__main__" :

    test_primitive_functions()
    my_str = "abcdefghijk"
    print( f"{my_str[:10]: <10} <--")
    my_str = "abcd"
    print( f"{my_str[:10]: <10} <--")
