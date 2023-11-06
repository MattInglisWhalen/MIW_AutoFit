from __future__ import annotations

# built-in libraries
import inspect
from typing import Callable, Union

# external libraries
import numpy as np

# internal classes
from autofit.src.package import logger


class PrimitiveFunction:

    """
    A wrapper for basic single-variable functions with one parameter

    Instance variables

    _name : string
    _func : a callable function with the form f(x, arg), e.g. def quad(x, arg): return arg*x^2
    _arg  : the argument for the function
    _callable_1_param : a callable function of the form f(x), e.g. np.exp(x). This is then converted into a _func

    Static Variables

    _built_in_prims_dict : a dictionary of all usable primitives

    """

    _built_in_prims_dict : dict[str,PrimitiveFunction] = {}

    def __init__(self, func: Union[None, Callable[[float,float],float]] = None,
                 name: str = "", arg : float = 1., other_callable: Union[None,Callable[[float],float]] = None):

        self._callable_1param: Union[None,Callable[[float],float]] = other_callable
        if func is None:
            if other_callable is not None :
                func = self.callable_2param
            else :
                func = PrimitiveFunction.pow1
        self._func: Callable[[float,float],float] = func  # valid functions must all be of the type f(x,arg)

        self._name : str = name
        if name == "" :
            self._name = self._func.__name__

        self._arg : float = arg

    def __repr__(self):
        if self._callable_1param is not None :
            return f"Prim {self._name} uses arg*{self._callable_1param.__name__}(x) with coefficient {self._arg}"
        return f"Prim {self._name} uses {self._func.__name__}(x,arg) with coefficient {self._arg}"


    @property
    def name(self) -> str:
        # Try to keep names less than 10 characters, so a composite function's tree looks good
        return self._name

    @property
    def func(self) -> Callable[[float,float],float]:
        return self._func
    @func.setter
    def func(self, other: Callable[[float,float],float]):
        self._func = other

    def callable_2param(self, x, arg: float) -> float :
        if self._callable_1param is None :
            return 0
        try :
            return arg*self._callable_1param(x)
        except ValueError :
            logger(self, x, arg)
            raise ValueError
        except TypeError :
            logger(inspect.getmodule(self._callable_1param),self, x, arg)
            raise TypeError

    @property
    def arg(self) -> float:
        return self._arg
    @arg.setter
    def arg(self, val):
        self._arg = val

    def eval_at(self,x):
        return self._func(x, self._arg)

    def eval_deriv_at(self,x):
        # simple symmetric difference
        delta = 1e-5
        return (self.eval_at(x+delta) - self.eval_at(x-delta) ) / (2*delta)

    def copy(self):
        new_prim = PrimitiveFunction( name=self._name, func=self._func, arg=self.arg)
        return new_prim

    """
    Static methods
    """

    @staticmethod
    def pow_neg1(x, arg: float) -> float:
        try:
            return arg/x
        except ZeroDivisionError :
            return 1e5
    @staticmethod
    def pow0(x, arg: float) -> float:
        return arg*x**0
    @staticmethod
    def pow1(x, arg: float) -> float:
        return arg*x
    @staticmethod
    def pow2(x, arg: float) -> float:
        return arg*x*x
    @staticmethod
    def pow3(x, arg: float) -> float:
        return arg*x*x*x
    @staticmethod
    def pow4(x, arg: float) -> float:
        return arg*x*x*x*x

    @staticmethod
    def my_sin(x, arg: float) -> float:
        return arg*np.sin(x)
    @staticmethod
    def my_cos(x, arg: float) -> float:
        return arg*np.cos(x)
    @staticmethod
    def my_exp(x, arg: float) -> float:
        return arg*np.exp(x)
    @staticmethod
    def my_log(x, arg: float) -> float:
        return arg*np.log(x)


    # special callables for composite functions
    @staticmethod
    def dim0_pow2(x,arg):  # for Gaussian Composite
        return -x**2/(2*arg**2) if arg > 0 else 1e5
    # dim 1 specials
    @staticmethod
    def pow1_shift(x,arg):  # for Sigmoid Composite
        return x-arg
    @staticmethod
    def exp_dim1(x,arg) -> float:  # for Sigmoid Composite
        #y = np.clip(x/arg, a_min=-709.78271289338399673, a_max=None)
        return np.exp(-x/arg) if arg > 0 else 0
    @staticmethod
    def n_exp_dim2(x,arg) -> float:  # for Normal Composite
        return np.exp(-x**2/(2*arg**2) )/np.sqrt(2*np.pi*arg**2) if arg > 0 else 1e5



    # arbitrary powers can be created using function composition (i.e using the CompositeFunction class)
    # For example x^1.5 = exp( 1.5 log(x) )

    @staticmethod
    def sum_(x, arg) -> float:  # shouldn't be added to built_in_dict
        return x + 0*arg


    @staticmethod
    def build_built_in_dict() -> dict[str,PrimitiveFunction]:

        # Powers
        prim_pow_neg1 = PrimitiveFunction(func=PrimitiveFunction.pow_neg1 )
        prim_pow0 = PrimitiveFunction(func=PrimitiveFunction.pow0 )
        prim_pow1 = PrimitiveFunction(func=PrimitiveFunction.pow1 )
        prim_pow2 = PrimitiveFunction(func=PrimitiveFunction.pow2 )
        prim_pow3 = PrimitiveFunction(func=PrimitiveFunction.pow3 )
        prim_pow4 = PrimitiveFunction(func=PrimitiveFunction.pow4 )
        prim_sum = PrimitiveFunction(func=PrimitiveFunction.sum_)

        PrimitiveFunction._built_in_prims_dict["pow_neg1"] = prim_pow_neg1
        PrimitiveFunction._built_in_prims_dict["pow0"] = prim_pow0
        PrimitiveFunction._built_in_prims_dict["pow1"] = prim_pow1
        PrimitiveFunction._built_in_prims_dict["pow2"] = prim_pow2
        PrimitiveFunction._built_in_prims_dict["pow3"] = prim_pow3
        PrimitiveFunction._built_in_prims_dict["pow4"] = prim_pow4
        PrimitiveFunction._built_in_prims_dict["sum"] = prim_sum

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

        # special
        prim_shift = PrimitiveFunction(func=PrimitiveFunction.pow1_shift)
        PrimitiveFunction._built_in_prims_dict["pow1_shift"] = prim_shift

        return PrimitiveFunction._built_in_prims_dict

    @staticmethod
    def built_in_list() -> list[PrimitiveFunction]:
        built_ins = []
        if not PrimitiveFunction._built_in_prims_dict :
            PrimitiveFunction.build_built_in_dict()
        for key, prim in PrimitiveFunction._built_in_prims_dict.items():
            built_ins.append( prim )
        return built_ins

    @staticmethod
    def built_in_dict() -> dict[str,PrimitiveFunction]:
        return PrimitiveFunction._built_in_prims_dict

    @staticmethod
    def built_in(key) -> PrimitiveFunction:
        if not PrimitiveFunction._built_in_prims_dict :
            PrimitiveFunction.build_built_in_dict()

        if key[:3] == "Pow":
            degree = int(key[3:])

            def func(x, arg):
                return arg * x**degree
            return PrimitiveFunction(func=func, name=f"Pow{degree}")

        return PrimitiveFunction._built_in_prims_dict[key]


