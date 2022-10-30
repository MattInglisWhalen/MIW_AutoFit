from __future__ import annotations

# built-in libraries
from typing import Callable

# external libraries
import numpy as np


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

    def __init__(self, func : Callable[[float],float] = None, name : str = "" ):

        if func is None :
            self._func = PrimitiveFunction.pow1
        else :
            self._func : Callable[[float],float] = func  # valid functions must all be of the type f(x,arg)

        if name == "" :
            self._name = self._func.__name__
        else :
            self._name: str = name



    def __repr__(self):
        return f"Primitive {self._name} uses {self._func.__name__}(x)"

    """
    Properties
    """

    @property
    def name(self):
        # Try to keep names less than 10 characters, so a composite function's tree looks good
        return self._name
    @name.setter
    def name(self, val):
        self._name = val

    @property
    def func(self):
        return self._func
    @func.setter
    def func(self, other):
        self._func = other

    def eval_at(self,x):
        return self._func(x)
    def eval_deriv_at(self,x):
        delta = 1e-5
        return (self.eval_at(x+delta) - self.eval_at(x-delta) ) / (2*delta)

    def copy(self):
        new_prim = PrimitiveFunction(name=self._name, func=self._func)
        return new_prim

    """
    Static methods
    """

    # positives
    @staticmethod
    def pow_neg1(x):
        try:
            return 1/x
        except ZeroDivisionError :
            print(f"pow_neg1: {x} is too small")
            return 1e10
    # noinspection PyUnusedLocal
    @staticmethod
    def pow0(x):
        return 1
    @staticmethod
    def pow1(x):
        return x
    @staticmethod
    def pow2(x):
        return x*x
    @staticmethod
    def pow3(x):
        return x*x*x
    @staticmethod
    def pow4(x):
        return x*x*x*x
    @staticmethod
    def my_sin(x):
        return np.sin(x)
    @staticmethod
    def my_cos(x):
        return np.cos(x)
    @staticmethod
    def my_exp(x):
        try :
            return np.exp(x)
        except RuntimeWarning :
            print(f"my_exp: {x} is large")
            return 1e10
    @staticmethod
    def my_log(x):
        try :
            return np.log(x*x)/2  # controversial choice
        except RuntimeWarning :
            print(f"my_log: {x} is too small")
            return 1e10



    # arbitrary powers can be created using function composition (i.e using the CompositeFunction class)
    # For example x^1.5 = exp( 1.5 log(x) )


    @staticmethod
    def build_built_in_dict():

        # Powers
        prim_pow0 = PrimitiveFunction(func=PrimitiveFunction.pow0 )
        prim_pow1 = PrimitiveFunction(func=PrimitiveFunction.pow1 )
        prim_pow2 = PrimitiveFunction(func=PrimitiveFunction.pow2 )
        prim_pow3 = PrimitiveFunction(func=PrimitiveFunction.pow3 )
        prim_pow4 = PrimitiveFunction(func=PrimitiveFunction.pow4 )
        prim_pow_neg1 = PrimitiveFunction(func=PrimitiveFunction.pow_neg1 )

        PrimitiveFunction._built_in_prims_dict["pow0"] = prim_pow0
        PrimitiveFunction._built_in_prims_dict["pow1"] = prim_pow1
        PrimitiveFunction._built_in_prims_dict["pow2"] = prim_pow2
        PrimitiveFunction._built_in_prims_dict["pow3"] = prim_pow3
        PrimitiveFunction._built_in_prims_dict["pow4"] = prim_pow4
        PrimitiveFunction._built_in_prims_dict["pow_neg1"] = prim_pow_neg1

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

