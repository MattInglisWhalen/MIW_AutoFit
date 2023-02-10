
# default libraries
import re as regex
from typing import Callable

# external libraries
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.special

from tkinter import Label as tk_label

# internal classes
from autofit.src.datum1D import Datum1D
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.composite_function import CompositeFunction
from autofit.src.package import logger

"""
This Optimizer class creates a list of possible fit functions. 
Using scipy's curve-fit function, each of the possible fit functions are fit to the data provided.
This class will then determine which of the functions is the best model for the data, using the reduced chi^2 statistic
as a discriminator. The model which achieves the lowest reduced chi-squared is chosen to be the "best" model
"""


class Optimizer:

    # noinspection PyTypeChecker
    def __init__(self, data=None, use_functions_dict = None, max_functions=3, regen=True, criterion="rchisqr"):

        logger("New optimizer created")

        # datasets, which are lists of Datum1D instances
        self._data : list[Datum1D] = []  # the raw datapoints of (x,y), possibly with uncertainties. It may
                                         # also represent a binned point if the input file is a list of x_values only

        # fit results
        self._shown_model : CompositeFunction = None          # a CompositeFunction
        self._shown_covariance : np.ndarray = None
        self._shown_rchisqr : float = 1e5

        # top 5 models and their data
        self._top5_models : list[CompositeFunction] = []
        self._top5_covariances : list[np.ndarray] = []
        self._top5_rchisqrs : list[float] = [1e5]  # [1e5  for _ in range(5)]

        # function construction parameters
        self._max_functions : int = max_functions
        self._primitive_function_list : list[PrimitiveFunction] = []
        self._primitive_names_list : list[str] = []

        # useful auxiliary varibles
        self._temp_function : CompositeFunction = None      # for ?
        self._cos_freq_list : list[float] = None
        self._sin_freq_list : list[float] = None
        self._cos_freq_list_dup : list[float] = None
        self._sin_freq_list_dup : list[float] = None

        if data is not None :
            self._data = sorted(data)  # list of Datum1D. Sort makes the data monotone increasing in x-value

        if use_functions_dict is None:
            use_functions_dict = {}
        self._use_functions_dict : dict[str,bool] = use_functions_dict
        self.load_default_functions()
        self.load_non_defaults_to_primitive_function_list()

        self._composite_function_list = []
        self._composite_generator = None
        self._gen_idx = -1

        self._criterion : Callable[[CompositeFunction],float] = None
        if criterion == "AIC" :
            self._criterion = self.Akaike_criterion
        elif criterion == "AICc" :
            self._criterion = self.Akaike_criterion_corrected
        elif criterion == "BIC" :
            self._criterion = self.Bayes_criterion
        elif criterion == "HQIC" :
            self._criterion = self.HannanQuinn_criterion
        else :
            self._criterion = self.reduced_chi_squared_of_fit

        self._regen_composite_flag = regen

    def __repr__(self):
        return f"Optimizer with {self._max_functions} m.f."

    @property
    def shown_model(self) -> CompositeFunction:
        return self._shown_model
    @shown_model.setter
    def shown_model(self, other):
        self._shown_model = other
    @property
    def shown_parameters(self) -> list[float]:
        return self._shown_model.args
    @shown_parameters.setter
    def shown_parameters(self, args_list):
        self._shown_model.args = args_list
    @property
    def shown_uncertainties(self) -> list[float]:
        return list(np.sqrt(np.diagonal(self._shown_covariance)))
    @property
    def shown_covariance(self):
        return self._shown_covariance
    @shown_covariance.setter
    def shown_covariance(self, cov_np):
        self._shown_covariance = cov_np
    @property
    def shown_rchisqr(self) -> float:
        return self._shown_rchisqr
    @shown_rchisqr.setter
    def shown_rchisqr(self, val):
        self._shown_rchisqr = val
    @property
    def shown_cc(self) -> np.ndarray:
        return coefficient_of_covariance(self.shown_covariance,self.shown_parameters)
    @property
    def shown_cc_offdiag_sumsqrs(self) -> float:
        return sum_sqr_off_diag( self.shown_cc )
    @property
    def avg_off(self) -> float:
        N = len(self.shown_parameters)
        return self.shown_cc_offdiag_sumsqrs / (N*(N-1)) if N > 1 else 0

    @property
    def top5_models(self) -> list[CompositeFunction]:
        return self._top5_models
    @top5_models.setter
    def top5_models(self, other):
        self._top5_models = other
    @property
    def top_model(self) -> CompositeFunction:
        return self.top5_models[0]
    @property
    def top5_args(self) -> list[list[float]]:
        return [model.args for model in self.top5_models]
    @property
    def top_args(self) -> list[float]:
        return self.top5_args[0]
    @property
    def top5_uncertainties(self) -> list[list[float]]:
        return [list(np.sqrt(np.diagonal(covariance))) for covariance in self._top5_covariances]
    @property
    def top_uncs(self) -> list[float]:
        return self.top5_uncertainties[0]
    @property
    def top5_covariances(self):
        return self._top5_covariances
    @top5_covariances.setter
    def top5_covariances(self, other):
        self._top5_covariances = other
    @property
    def top_cov(self):
        return self.top5_covariances[0]
    @property
    def top5_names(self) -> list[str]:
        return [model.name for model in self._top5_models]
    @property
    def top_name(self) -> str:
        return self.top5_names[0]
    @property
    def top5_rchisqrs(self) -> list[float]:
        return self._top5_rchisqrs
    @top5_rchisqrs.setter
    def top5_rchisqrs(self, other):
        self._top5_rchisqrs = other
    @property
    def top_rchisqr(self):
        return self.top5_rchisqrs[0]

    @property
    def prim_list(self) -> list[PrimitiveFunction]:
        return self._primitive_function_list
    def remove_named_from_prim_list(self, name: str):
        for prim in self.prim_list[:] :
            if prim.name == name :
                self._primitive_function_list.remove(prim)
                return
        else :
            logger(f"No prim named {name} in optimizer prim list")

    @property
    def criterion(self) -> Callable[[CompositeFunction],float]:
        return self._criterion
    @criterion.setter
    def criterion(self,other: Callable[[CompositeFunction],float]):
        # logger(f"Changed {self._criterion} to {other}")
        self._criterion = other
        self.update_top5_rchisqrs_for_new_data(self._data)

    @property
    def composite_function_list(self):
        return self._composite_function_list
    @composite_function_list.setter
    def composite_function_list(self, comp_list):
        self._composite_function_list = comp_list
    @property
    def gen_idx(self) -> int:
        return self._gen_idx

    def update_opts(self, use_functions_dict: dict[str,bool], max_functions: int):

        self._use_functions_dict = use_functions_dict
        self._max_functions = max_functions

        self._primitive_function_list = []
        self.load_default_functions()
        self.load_non_defaults_to_primitive_function_list()

        self._regen_composite_flag = True

    # def update_top5_rchisqrs_for_new_data_single_model(self, new_data, chosen_model):
    #     self.set_data_to(new_data)
    #     for idx, model in enumerate(self.top5_models) :
    #         if model.name == chosen_model.name :
    #             self.top5_rchisqrs[idx] = self.reduced_chi_squared_of_fit(model)
    def update_top5_rchisqrs_for_new_data(self, new_data):
        logger(f"Updating top5 criterions. Before: {self.top5_rchisqrs}")
        self.set_data_to(new_data)
        for idx, model in enumerate(self.top5_models) :
            self.top5_rchisqrs[idx] = self.criterion(model)
        logger(f"After: {self.top5_rchisqrs}")

    # changes top5 lists, does not change _shown variables
    def query_add_to_top5(self, model: CompositeFunction, covariance):

        rchisqr = self.criterion(model)
        logger(f"Querying {rchisqr:.2F} for {model.name}")
        if np.isnan(rchisqr) or rchisqr > self.top5_rchisqrs[-1] or any( V < 0 for V in np.diagonal(covariance)) :
            return

        rchisqr_adjusted = self.criterion_w_cov_punish(model, covariance)
        if rchisqr_adjusted > rchisqr :
            logger(f"Adjustment: {rchisqr} -> {rchisqr_adjusted}")
            rchisqr = rchisqr_adjusted
        # check for duplication
        for idx, (topper, chisqr) in enumerate(zip( self.top5_models[:],self.top5_rchisqrs[:] )):
            # comparing with names
            if model.longname == topper.longname:
                logger("Same name in query")
                if chisqr <= rchisqr+1e-5 :
                    return
                # delete the original entry from the list because ???
                del self.top5_models[idx]
                del self.top5_covariances[idx]
                del self.top5_rchisqrs[idx]
                break
            if abs(rchisqr - chisqr) < 1e-5 :
                # essentially the same function, need to choose which one is the better representative

                # dof should trump all
                if topper.dof < model.dof :
                    logger(f"Booting out contender {model.name} with dof {model.dof} "
                           f"in favour of {topper.name} with dof {topper.dof}")
                    return
                # depth should then be preferred
                if topper.depth > model.depth :
                    logger(f"Booting out contender {model.name} with depth {model.width} "
                           f"in favour of {topper.name} with depth {topper.width}")
                    return
                # width should then be minimized
                if topper.width < model.width :
                    logger(f"Booting out contender {model.name} with width {model.width} "
                           f"in favour of {topper.name} with width {topper.width}")
                    return
                if topper.dof == model.dof and topper.depth == model.depth and topper.width == model.width :
                    logger(f"Booting out contender {model.name} in favour of {topper.name}, "
                           f"both with with dof, depth, width {topper.dof} {topper.depth} {topper.width}")
                    # default is to keep the first one added
                    return
                # more sophisticated distinguishers might also try to minimize correlations between parameters
                logger(f"Booting out {topper.name} in favour of contender {model.name}")
                del self.top5_models[idx]
                del self.top5_covariances[idx]
                del self.top5_rchisqrs[idx]
                break
                # after dof, depth should trump all


        # logger(f"Passed basic check, trying to add {rchisqr} to {self._top5_rchisqrs}")
        for idx, chi_sqr in enumerate(self._top5_rchisqrs[:]) :
            if rchisqr < self._top5_rchisqrs[idx] :
                self.top5_models.insert(idx, model.copy())
                self.top5_covariances.insert(idx, covariance)
                self.top5_rchisqrs.insert(idx, rchisqr)

                self.top5_models = self._top5_models[:5]
                self.top5_covariances = self._top5_covariances[:5]
                self.top5_rchisqrs = self._top5_rchisqrs[:5]

                par_str_list = [f"{arg:.3F}" for arg in model.args]
                unc_str_list = [f"{unc:.3F}" for unc in std(covariance)]

                logger(f"New top {idx} with red_chisqr={rchisqr:.2F}: "
                       f"{model=} with pars=[" + ', '.join(par_str_list) + "] \u00B1 [" + ', '.join(unc_str_list) + "]")
                model.print_tree()

                # also check any parameters for being "equivalent" to zero. If so, remove the d.o.f.
                # and add the new function to the top5 list
                for ndx, (arg, unc) in enumerate( zip(model.args,std(covariance)) ) :
                    if not np.isinf(unc) and abs(arg) < 2*unc :  # 95% confidence level arg is not different from zero

                        reduced_model = model.submodel_without_node_idx(ndx)
                        if ( reduced_model is None
                                or self.fails_rules(reduced_model)
                                or (reduced_model.prim.name == "sum_" and reduced_model.num_children() == 0) ):
                            logger("Zero arg detected but but can't reduce the model")
                            continue
                        logger("Zero arg detected: new trimmed model is")
                        reduced_model.set_submodel_of_zero_idx(model,ndx)
                        reduced_model.print_tree()
                        if reduced_model.name in self.top5_names :
                            reduced_idx = self.top5_names.index(reduced_model.name)
                            if reduced_idx < idx :
                                logger("Shazam")
                                # don't keep the supermodel
                                del self.top5_models[idx]
                                del self.top5_covariances[idx]
                                del self.top5_rchisqrs[idx]
                                break
                        # this method changes shown models
                        improved_reduced, improved_cov = self.fit_this_and_get_model_and_covariance(
                                                            model_=reduced_model,initial_guess=reduced_model.args,
                                                            change_shown = False
                                                         )
                        self.query_add_to_top5(model=improved_reduced, covariance=improved_cov)
                        break
                return

    # update for which reason?
    # def update_resort_top5_rchisqr(self):
    #     self._top5_rchisqrs = [1e5 for _ in self._top5_rchisqrs]
    #     for model, cov in zip(self._top5_models[:], self._top5_covariances[:]) :
    #         self.query_add_to_top5(model=model,covariance=cov)


    def composite_function_generator(self, depth, regen_built_ins = True):

        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        if depth == 0 :
            if regen_built_ins :
                logger("Composite generator:",self._primitive_names_list)
                logger("Starting new generator at 0 depth")
                self._gen_idx = 0
                for model in CompositeFunction.built_in_list() :
                    yield model

            for iprim in self._primitive_function_list:
                new_comp = CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                             children_list=[iprim])
                yield new_comp

        else :
            head_gen = self.composite_function_generator( depth = depth-1 , regen_built_ins=False)
            for icomp in head_gen :
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list:

                        # sums
                        new_comp = icomp.copy()
                        sum_node = new_comp.get_node_with_index(idescendent)
                        if sum_node.num_children() > 0 and iprim.name > sum_node.children_list[-1].prim.name  :
                            pass
                        else:
                            if sum_node.prim.name in ["my_sin","my_cos"] and iprim.name in ["my_sin","my_cos"] :
                                pass  # speedup for double trig
                            elif sum_node.prim.name == "my_log" and iprim.name in ["my_log","my_exp"] :
                                pass
                            elif sum_node.prim.name == "my_exp" and iprim.name in ["my_exp","pow0"] :
                                pass
                            else:
                                sum_node.add_child(iprim, update_name=True)
                                yield new_comp

                        # factors
                        new_mul = icomp.copy()
                        mul_node = new_mul.get_node_with_index(idescendent)
                        if mul_node.prim.name >= iprim.name :
                            if mul_node.prim.name == "my_exp" and iprim.name in ["my_exp","pow0"] :
                                pass  # speedup for multiplied exps
                            else :
                                mul_node.add_younger_brother(iprim, update_name=True)
                                yield new_mul
    def valid_composite_function_generator(self, depth):

        all_comps_at_depth = self.composite_function_generator(depth)
        for icomp in all_comps_at_depth :
            if not self.fails_rules(icomp):
                yield icomp
    def all_valid_composites_generator(self):
        for idepth in range(7):
            for icomp in self.valid_composite_function_generator(depth=idepth):
                self._gen_idx += 1
                yield icomp

    # TODO: make a generator version of this e.g. [] -> ()
    def build_composite_function_list(self, status_bar : tk_label):
        # the benefit of using this is that you can generate it once, and if the options don't change you don't need
        # to generate it again. Follow that logic
        if not self._regen_composite_flag :
            return
        logger(f"{self._regen_composite_flag}, so regenerating composite list with {self._max_functions} "
               f"and {[prim.name for prim in self._primitive_function_list]}")
        self._regen_composite_flag = False

        # start with simple primitives in a sum
        self._composite_function_list = []
        for iprim in self._primitive_function_list:
            new_comp = CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                         children_list=[iprim])
            self._composite_function_list.append(new_comp)

        last_list = self._composite_function_list
        for depth in range(self._max_functions-1) :
            new_list = []
            for icomp in last_list:
                status_bar.configure(text=f"   Stage 1/3: {len(last_list)+len(new_list):>10} naive models generated,"
                                          f" {0:>10} models fit.")
                status_bar.master.master.update()
                if status_bar['bg'] == "#010101":  # cancel code
                    self._regen_composite_flag = True
                    break
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list[:]:

                        new_comp = icomp.copy()
                        sum_node = new_comp.get_node_with_index(idescendent)
                        # the naming already sorts the multiplication and summing parts by descending order,
                        # so why not just build that in
                        if sum_node.num_children() > 0 and iprim.name > sum_node.children_list[-1].prim.name  :
                            pass
                        else:
                            if sum_node.prim.name in ["my_sin","my_cos"] and iprim.name in ["my_sin","my_cos"] :
                                pass  # speedup for double trig
                            elif sum_node.prim.name == "my_log" and iprim.name in ["my_log","my_exp"] :
                                pass
                            elif sum_node.prim.name == "my_exp" and iprim.name in ["my_exp","pow0"] :
                                pass
                            else:
                                sum_node.add_child(iprim, update_name=True)
                                new_list.append(new_comp)

                        new_mul = icomp.copy()
                        mul_node = new_mul.get_node_with_index(idescendent)
                        if mul_node.prim.name >= iprim.name :
                            if mul_node.prim.name == "my_exp" and iprim.name in ["my_exp","pow0"] :
                                pass  # speedup for multiplied exps
                            else :
                                mul_node.add_younger_brother(iprim, update_name=True)
                                new_list.append(new_mul)
                if status_bar['bg'] == "#010101" :  # cancel code
                    break
            logger(f"{depth} build_comp_list new_len=",len(new_list))
            self._composite_function_list.extend( new_list )
            last_list = new_list

        for comp in CompositeFunction.built_in_list():
            self._composite_function_list.append(comp.copy())

        # prepend the current top 5 models

        self.trim_composite_function_list(status_bar=status_bar)
        logger(f"After trimming list: (len={len(self._composite_function_list)})")
        for icomp in self._composite_function_list:
            logger(icomp)
        logger("|----------------\n")
    def trim_composite_function_list(self,status_bar:tk_label):
        # Finds and removes duplicates
        # Performs basic algebra to recognize simpler forms with fewer parameters
        # E.g. Functions like pow0(pow0) and pow1(pow1 + pow1) have fewer arguments than the naive calculation expects

        # Only run this at the end of the tree generation: if you run this in intermediate steps,
        # you will miss out on some functions possibilities

        num_comps = len(self._composite_function_list[:])
        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        # use regex to trim based on rules applied to composite names
        for idx, icomp in enumerate(self._composite_function_list[:]) :

            if idx % 500 == 0 :
                status_bar.configure(text=f"   Stage 2/3: {len(self._composite_function_list):>10} valid "
                                          f"models generated, {0:>10} models fit.")
                status_bar.master.master.update()
                if status_bar['bg'] == "#010101" :  # cancel code
                    self._regen_composite_flag = True
                    break
                logger(f"{idx}/{num_comps}")

            # if self.fails_rules(icomp) :
            if self.validate_fails(icomp) :
                self._composite_function_list.remove(icomp)

        # remove duplicates using a dict{}
        self._composite_function_list = list(
            { icomp.__repr__() : icomp for icomp in self._composite_function_list[:] }.values()
        )
    def fails_rules(self, icomp):

        name = icomp.name

        """
        I imagine non-regex is faster than regex
        """

        if icomp.prim.name == "sum_" and icomp.num_children() < 2 and icomp.younger_brother is not None :
            return 1
        # trig inside trig is too complex, and very (very) rarely occurs in applications
        if icomp.has_double_trigness() or icomp.has_double_expness() or icomp.has_double_logness():
            return 29
        # sins, cosines, exps, and logs never have angular frequency or decay parameters exactly 1
        # all unitless-argument functions start with my_
        if icomp.has_argless_explike() :
            return 43
        # pow1 with composition of exactly one term
        if icomp.has_trivial_pow1():
            return 13*17
        # pow1 used as a sub-sum inside head sum
        for child in icomp.children_list:
            if (child.prim.name == "pow1" and child.num_children() > 0
                    and child.younger_brother is None and child.older_brother is None) :
                return 97

        # repeated reciprocal is wrong
        if icomp.has_repeated_reciprocal() :
            return 7
        # pow1 times pow1,2,3,4 is wrong
        if icomp.has_reciprocal_cancel_pospow() :
            return 61
        # if regex.search(f"pow_neg1\(pow_neg1\)", name):
        #     return 7
        #     # and deeper
        # if regex.search(f"pow_neg1\(pow_neg1\([a-z0-9_+]*]\)\)", name):
        #     return 7
        if icomp.has_log_with_odd_power() :
            return 67
        # composition of powers with no sum is wrong -- not strictly true -- think of log^2(Ax+B)
        # if regex.search(f"pow[0-9]\(pow[a-z0-9_·]*\)", name):
        #     return 5
        if icomp.has_pow_with_no_sum():
            return 5


        # composition of a constant function is wrong
        if regex.search(f"pow0\(", name):
            return 2
        if regex.search(f"\(pow0\)", name):
            return 3
        # composition of powers with no sum is wrong -- not strictly true -- think of log^2(Ax+B)
        if regex.search(f"pow[0-9]\(pow[a-z0-9_·]*\)", name):
            return 5
        # trivial reciprocal is wrong
        if regex.search(f"pow_neg1\(pow1\)", name):
            return 11

        # sum of the same function (without further composition) is wrong
        for prim_name in self._primitive_names_list:
            if regex.search(f"{prim_name}[a-z0-9_+]*{prim_name}", name):
                return 23

        # pow0+log(...) is a duplicate since A + Blog( Cf(x) ) = B log( exp(A/B) Cf(x) ) = log(f)
        if regex.search(f"pow0\+my_log\([a-z0-9_]*\)", name) \
                or regex.search(f"my_log\([a-z0-9_]*\)\+pow0", name):
            return 37

        # more more exp algebra: Alog( Bexp(Cx) ) = AlogB + ACx = pow0+pow1
        # if regex.search(f"my_log\(my_exp\(pow1\)\)", name):
        #     raise EnvironmentError
        #     return 53



        # log(1/f+g) or log( (f+g)^n )is the same as log(f+g) -- again, not strictly true
        # if regex.search(f"my_log\(pow[a-z0-9_]*\([a-z0-9_+]*\)\)", name):
        #     return 73

        # pow3(exp(...)) is just exp(3...)
        if regex.search(f"pow[a-z0-9_]*\(my_exp\([a-z0-9_+]*\)\)", name):
            return 83

        """
        Multiplicative rules
        """
        if regex.search(f"pow0·", name) or regex.search(f"·pow0", name):
            return 1000 + 2

        return 0

    def validate_fails(self, icomp):

        name = icomp.name
        remove_flag = self.fails_rules(icomp)

        # second check on the fails we did or didnt experience
        good_list = ["my_exp(my_log)",
                     "pow1(my_cos(pow1)+my_sin(pow1))",
                     "pow_neg1",
                     "sum_(pow0)",
                     "sum_(pow1)",
                     "pow1",
                     "pow0",
                     "pow1·pow1",
                     "pow1·sum_(pow0+pow1·pow1)",
                     "pow1·pow1(pow0+pow1)",
                     "my_exp(my_log)+my_exp(my_log)",
                     # "my_exp(pow2(my_log))",
                     "my_sin(pow0+my_exp(pow1)),"
                     "pow4+pow1"]

        for good in good_list :
            if name == good and remove_flag:
                logger(f"\n\n>>> Why did we remove {name=} at {remove_flag=} <<<\n\n")
                raise SystemExit

        if name == "pow1(pow0+pow0)" and not remove_flag:
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=23 <<<\n\n")
            raise SystemExit

        if name == "pow1·pow1(pow1·pow1)" and not remove_flag:
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=?? <<<\n\n")
            raise SystemExit

        if name == "my_exp(pow1)·my_exp·pow1" and not remove_flag:
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=43 <<<\n\n")
            logger(icomp.has_double_expness())
            raise SystemExit

        if (name == "pow1·my_exp(my_exp)" or name == "pow1·my_exp(pow1·my_exp)") and not remove_flag:
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=29 <<<\n\n")
            logger(icomp.has_double_expness())
            raise SystemExit

        if name == "my_exp(pow1)·my_exp(pow1)" :
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=?? <<<\n\n")
            raise RuntimeWarning

        return remove_flag

    def add_primitive_to_list(self, name: str, functional_form: str) -> str:
        # For adding one's own Primitive Functions to the built-in list

        if self._use_functions_dict["custom"] != 1 :
            return ""
        if name in [prim.name for prim in self._primitive_function_list ] :
            return ""

        # logger(f"\n{name} {functional_form}")
        if regex.search("\\\\",functional_form) or regex.search("\n",functional_form)\
                or regex.search("\s",functional_form) :
            return "Stop trying to inject code"


        code_str  = f"def {name}(x,arg):\n"
        code_str += f"    return arg*({functional_form})\n"
        code_str += f"new_prim = PrimitiveFunction(func={name})\n"
        code_str += f"dict = PrimitiveFunction.built_in_dict()\n"
        code_str += f"dict[\"{name}\"] = new_prim\n"

        try:
            exec(code_str)
        except SyntaxError :
            return f"Corrupted custom function {name} " \
                   f"with form={functional_form}, returning to blank slate."

        logger("add_primitive_to...")
        for key, val in PrimitiveFunction.built_in_dict().items() :
            logger(f"{key}, {val}")

        try :
            PrimitiveFunction.built_in(name).eval_at(np.pi/4)
        except NameError :
            return f"One of the functions used in your custom function {name} \n" \
                   f"    with form {functional_form} does not exist."

        self._primitive_function_list.append( PrimitiveFunction.built_in(name) )

        return ""
    def set_data_to(self, other_data):
        self._data = sorted(other_data)

    # does not change self._data
    def fit_setup(self, fourier_condition = True) -> tuple[list[float], list[float], list[float], bool]:

        x_points = []
        y_points = []
        sigma_points = []

        use_errors = True
        y_range = max([datum.val for datum in self._data]) - min([datum.val for datum in self._data])
        for datum in self._data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            if datum.sigma_val < 1e-10:
                use_errors = False
                sigma_points.append(y_range / 10)
                # datum.sigma_val = y_range/10  # if you want chi_sqr() to work for zero-error data, you need this
            else:
                sigma_points.append(datum.sigma_val)

        # do an FFT if there's a sin/cosine --
        # we zero-out the average height of the data to remove the 0-frequency mode as a contribution
        # then to pass in the dominant frequencies as arguments to the initial guess
        if fourier_condition :
            self.create_cos_sin_frequency_lists()

        return x_points, y_points, sigma_points, use_errors

    # does not change input model_
    # changes _shown variables
    # does not change top5 lists
    def fit_loop(self, model_, x_points, y_points, sigma_points, use_errors,
                       initial_guess = None, info_string = ""):

        model = model_.copy()

        if model.num_trig() > 0 and initial_guess is None:
            self._cos_freq_list_dup = self._cos_freq_list.copy()
            self._sin_freq_list_dup = self._sin_freq_list.copy()

        logger(f"\nFitting {model=}")
        logger(model.tree_as_string_with_dimensions())

        # Find an initial guess for the parameters based off scaling arguments
        if initial_guess is None :
            if "Pow" not in model.name :
                initial_guess = self.find_initial_guess_scaling(model)
            else:
                degree = model.name.count('+')
                np_args = np.polyfit(x_points,y_points,degree)
                leading = np_args[0]
                trailing = np_args[1:]/leading
                initial_guess = [leading] + list(trailing)

        logger(f"{info_string}Scaling guess: {initial_guess}")

        # Next, find a better guess by relaxing the error bars on the data
        # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
        # and so relaxed error bars help point towards global minima
        try:
            better_guess, better_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                                 p0=initial_guess, maxfev=5000, method='lm')
            if any( [x < 0 for x in np.diagonal(better_cov)] ) :
                logger("Negative variance encountered")
                raise RuntimeError
        except RuntimeError:
            logger("Couldn't find optimal parameters for better guess.")
            model.args = list(initial_guess)
            return model, np.array([1e10 for _ in range(len(initial_guess)**2)]) \
                            .reshape(len(initial_guess),len(initial_guess))
        logger(f"{info_string}Better guess: {better_guess} +- {np.sqrt(np.diagonal(better_cov))}")

        # Finally, use the better guess to find the true minimum with the true error bars
        try:
            np_pars, np_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                        sigma=sigma_points, absolute_sigma=use_errors,
                                        p0=better_guess, maxfev=5000)
            if any([x < 0 for x in np.diagonal(np_cov)]):
                logger("Negative variance encountered")
                raise RuntimeError
        except RuntimeError:
            logger("Couldn't find optimal parameters for final fit.")
            model.args = list(better_guess)
            return model, better_cov
        logger(f"{info_string}Final guess: {np_pars} +- {np.sqrt(np.diagonal(np_cov))}")
        model.args = list(np_pars)

        # should use the effective variance method if x-errors exist, e.g.
        # sigma^2 = sigma_y^2 + sigma_x^2 (dy/dx)^2
        if any( [datum.sigma_pos**2 / (datum.pos**2 + 1e-10) > 1e-10 for datum in self._data] ):
            for it in range(3):  # iterate 3 times
                effective_sigma = np.sqrt( [datum.sigma_val**2 + datum.sigma_pos**2 * model.eval_deriv_at(datum.pos)**2
                                            for datum in self._data] )
                try:
                    np_pars, np_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                                        sigma=effective_sigma, absolute_sigma=use_errors,
                                                        p0=np_pars, maxfev=5000)
                except RuntimeError :
                    logger(f"On model {model} max_fev reached")
                    # raise RuntimeError
                model.args = list(np_pars)
                logger(f"Now with effective variance: {np_pars} +- {np.sqrt(np.diagonal(np_cov))}")

        return model, np_cov

    # does not change input model_
    # changes _shown variables, unless toggled off
    # does not change top5 lists
    def fit_this_and_get_model_and_covariance(self, model_ : CompositeFunction,
                                                    initial_guess = None, change_shown = True,
                                                    do_halving = False, halved = 0):

        model = model_.copy()
        init_guess = initial_guess
        if init_guess is None:
            while model.is_submodel :
                supermodel = model_.submodel_of.copy()
                logger(f"\n>>> {model_}: Have to go back to the supermodel {supermodel} "
                       f"for refitting. {model_.submodel_zero_index}")
                supermodel, _ = self.fit_this_and_get_model_and_covariance(model_=supermodel, change_shown=False,
                                                                           do_halving=do_halving, halved=halved)
                logger(f"rchisqr for supermodel= {self.reduced_chi_squared_of_fit(supermodel)} {do_halving=} {halved=}")
                model = supermodel.submodel_without_node_idx(model_.submodel_zero_index)
                # ^ creates a submodel that doesnt think it's a submodel
                init_guess = model.args
        if model.name[-8:] == "Gaussian" :
            modal = model.num_children()
            if modal > 1 :
                means = self.find_peaks_for_gaussian(expected_n = modal)
                logger(f"Peaks expected at {means}")
                widths = self.find_widths_for_gaussian(means=means)
                logger(f"Widths expected to be {widths}")
                est_amplitudes = self.find_amplitudes_for_gaussian(means=means,widths=widths)
                logger(f"Amplitudes expected to be {est_amplitudes}")
                init_guess = []
                for amp, width, mean in zip(est_amplitudes, widths, means) :
                    init_guess.extend( [amp,width,mean] )

        x_points, y_points, sigma_points, use_errors = self.fit_setup()

        fitted_model, fitted_cov = self.fit_loop(model_=model, initial_guess=init_guess,
                                                 x_points=x_points,y_points=y_points,sigma_points=sigma_points,
                                                 use_errors=use_errors)
        fitted_rchisqr = self.reduced_chi_squared_of_fit(fitted_model)

        if fitted_rchisqr > 10 and len(self._data) > 20 and do_halving:
            logger(" "*halved*4 + f"It is unlikely that we have found the correct model... "
                                 f"halving the dataset to {len(self._data)//2}")

            lower_data = self._data[:len(self._data)//2]
            lower_optimizer = Optimizer(data=lower_data)
            lower_optimizer.fit_this_and_get_model_and_covariance(model_=model,change_shown=True,
                                                                  do_halving=True, halved=halved + 1)
            lower_rchisqr = self.reduced_chi_squared_of_fit(lower_optimizer.shown_model)
            if lower_rchisqr < fitted_rchisqr :
                fitted_model = lower_optimizer.shown_model
                fitted_cov = lower_optimizer.shown_covariance
                fitted_rchisqr = lower_rchisqr

            upper_data = self._data[len(self._data)//2:]
            upper_optimizer = Optimizer(data=upper_data)
            upper_optimizer.fit_this_and_get_model_and_covariance(model_=model,change_shown=True,
                                                                  do_halving=True, halved=halved + 1)
            upper_rchisqr = self.reduced_chi_squared_of_fit(upper_optimizer.shown_model)
            if lower_rchisqr < fitted_rchisqr :
                fitted_model = upper_optimizer.shown_model
                fitted_cov = upper_optimizer.shown_covariance
                fitted_rchisqr = upper_rchisqr

        if initial_guess is None and model_.is_submodel :
            # make the submodel realize it's a submodel
            fitted_model.set_submodel_of_zero_idx(model_.submodel_of, model_.submodel_zero_index)

        if change_shown :
            self._shown_model = fitted_model
            self._shown_covariance = fitted_cov
            self._shown_rchisqr = fitted_rchisqr

        return fitted_model, fitted_cov

    # changes top5
    # does not change _shown variables
    def find_best_model_for_dataset(self, status_bar : tk_label, halved=0):

        if not halved :
            self.build_composite_function_list(status_bar=status_bar)

        x_points, y_points, sigma_points, use_errors = self.fit_setup()

        num_models = len(self._composite_function_list)
        for idx, model in enumerate(self._composite_function_list) :

            fitted_model, fitted_cov = self.fit_loop(model_=model,
                                                     x_points=x_points, y_points=y_points, sigma_points=sigma_points,
                                                     use_errors=use_errors,info_string=f"{idx+1}/{num_models} ")

            self.query_add_to_top5(model=fitted_model, covariance=fitted_cov)
            status_bar.configure(text=f"   Stage 3/3: {len(self._composite_function_list):>10} valid models generated,"
                                      f" {idx+1:>10} models fit.")
            status_bar.master.master.update()
            if status_bar['bg'] == "#010101" :  # cancel code
                break
            # if model.name == "my_exp(my_sin(pow1))" :
            #     logger(fitted_model.args)
            #     logger(self._sin_freq_list)
            #     logger([2*np.pi*freq for freq in self._sin_freq_list])
            #     logger(self._cos_freq_list)
            #     logger([2*np.pi*freq for freq in self._cos_freq_list])
            #
            #     self.show_fit(model=fitted_model,pause_on_image=True)



        logger(f"\nBest models are {[m.name for m in self.top5_models]} with "
               f"associated reduced chi-squareds {self.top5_rchisqrs}")

        logger(f"\nBest model is {self.top_model} "
               f"\n with args {self.top_args} += {self.top_uncs} "
               f"\n and reduced chi-sqr {self.top_rchisqr}")
        self.top_model.print_sub_facts()
        self.top_model.print_tree()

        if self.top_rchisqr > 10 and len(self._data) > 20:
            # if a better model is found here it will probably underestimate the actual rchisqr since
            # it will only calculate based on half the data
            logger("It is unlikely that we have found the correct model... halving the dataset")
            lower_data = self._data[:len(self._data)//2]
            lower_optimizer = Optimizer(data=lower_data,
                                        use_functions_dict=self._use_functions_dict,
                                        max_functions=self._max_functions, regen=False)
            lower_optimizer.composite_function_list = self._composite_function_list
            lower_optimizer.find_best_model_for_dataset(status_bar=status_bar)
            for l_model, l_cov in zip(lower_optimizer.top5_models,lower_optimizer.top5_covariances) :
                self.query_add_to_top5(model=l_model,covariance=l_cov)

            upper_data = self._data[len(self._data)//2:]
            upper_optimizer = Optimizer(data=upper_data,
                                        use_functions_dict=self._use_functions_dict,
                                        max_functions=self._max_functions, regen=False)
            upper_optimizer.composite_function_list = self._composite_function_list
            upper_optimizer.find_best_model_for_dataset(status_bar=status_bar)
            for u_model, u_cov in zip(upper_optimizer.top5_models,upper_optimizer.top5_covariances) :
                self.query_add_to_top5(model=u_model,covariance=u_cov)

    # changes top5
    # does not change _shown variables
    def async_find_best_model_for_dataset(self, start=False) -> str:

        status = ""
        if start:
            self._composite_generator = self.all_valid_composites_generator()

        x_points, y_points, sigma_points, use_errors = self.fit_setup(fourier_condition=start)

        batch_size = 10
        for idx in range(batch_size) :

            try:
                model = next(self._composite_generator)
            except StopIteration :
                status = "Done: stop iteraton reach"
                break

            fitted_model, fitted_cov = self.fit_loop(model_=model,
                                                     x_points=x_points, y_points=y_points, sigma_points=sigma_points,
                                                     use_errors=use_errors, info_string=f"Async {self._gen_idx+1} ")

            self.query_add_to_top5(model=fitted_model, covariance=fitted_cov)


        # self.shown_model = self.top5_models[0]
        # self.shown_covariance = self.top5_covariances[0]
        # self.shown_rchisqr = self.top5_rchisqrs[0]

        return status

    def find_peaks_for_gaussian(self, expected_n):
        # we assume that the peaks are "easy" to find -- that there is a zero-derivative at each one

        # there should be at least 3*expected_n datapoints
        smoothed = self.smoothed_data(n=2)   # points -= 2
        if len(self._data) > 6 :
            smoothed = self.smoothed_data(n=3)  # points -= 3
        slope = self.deriv_n(data=smoothed,n=1)  # points -= 1
        if len(self._data) > 7 :
            slope = self.smoothed_data(data=slope,n=1)  # points -= 1
        # so in the worst case with npoints = 6, we still have 3 points to work with

        cand_con = []
        for m0, m1 in zip( slope[:-1], slope[1:] ) :
            if np.sign(m0.val) != np.sign(m1.val) :
                tup = (m0.pos + m1.pos)/2, (m1.val-m0.val)/(m1.pos-m0.pos)
                cand_con.append( tup )
                logger(f"New candidate at {(m0.pos + m1.pos)/2} "
                       f"with concavity {(m1.val-m0.val)/(m1.pos-m0.pos)}")

        # sort the candidates by their concavity
        sorted_candidates = [cc[0] for cc in cand_con]
        if len(cand_con) > expected_n :
            sorted_cand_con = sorted(cand_con, key=lambda x: x[1] )
            sorted_candidates = [ cc[0] for cc in sorted_cand_con ]
        elif len(cand_con) < expected_n :
            for _ in range( expected_n-len(cand_con) ) :
                sorted_candidates.append( (self._data[0].pos+self._data[-1].pos)/2 )

        return sorted_candidates[:expected_n]
    def find_widths_for_gaussian(self, means) :
        # for mean i, guess that the width is half the distance to the nearest other peak
        widths = []
        avg_bin_width = (self._data[-1].pos - self._data[0].pos) / ( len(self._data) - 1)
        for mean in means :
            distances = [ abs(mean-x) for x in means ]
            nearest = sorted( distances )[1]  # [0] will always be a zero distance
            widths.append( nearest/2 if nearest > avg_bin_width else avg_bin_width )
        return widths
    def find_amplitudes_for_gaussian(self, means, widths) :
        amplitudes = []
        for mean in means :
            for datumlow, datumhigh in zip( self._data[:-1], self._data[1:] ) :
                if datumlow.pos <= mean < datumhigh.pos :
                    amplitudes.append( (datumlow.val+datumhigh.val)/2 )
                    break
        avg_bin_width = (self._data[-1].pos - self._data[0].pos) / ( len(self._data) - 1)
        expected_amplitude_sum = sum( [datum.val for datum in self._data] )*avg_bin_width
        actual_amplitude_sum = sum( amplitudes )
        return [ amp*expected_amplitude_sum/actual_amplitude_sum/np.sqrt(2*np.pi*width**2)
                 for (amp,width) in zip(amplitudes,widths) ]

    def create_cos_sin_frequency_lists(self):

        self._cos_freq_list = []
        self._sin_freq_list = []

        # need to recreate the data as a list of uniform x-interval pairs
        # (in case the data points aren't uniformly spaced)
        minx = min( [datum.pos for datum in self._data] )
        maxx = max( [datum.pos for datum in self._data] )
        if (maxx-minx) < 1e-10 :
            self._cos_freq_list.append(0)
            self._sin_freq_list.append(0)
            return
        x_points = np.linspace(minx, maxx, num=len(self._data)+1)  # supersample of the data, with endpoints
                                                                   # the same and one more for each interval

        # very inefficient
        y_points = [ self._data[0].val ]
        for target in x_points[1:-1] :
            # find the two x-values in _data that surround the target value
            for idx, pos_high in enumerate( [datum.pos for datum in self._data] ):
                if pos_high > target :
                    # equation of the line connecting the two points is
                    # y(x) = [ y_low(x_high-x) + y_high(x-x_low) ] / (x_high - x_low)
                    x_low, y_low = self._data[idx-1].pos, self._data[idx-1].val
                    x_high, y_high = self._data[idx].pos, self._data[idx].val  # wtf its been wrong this entire time
                    y_points.append( ( y_low*(x_high-target) + y_high*(target-x_low) ) / (x_high - x_low)  )
                    break

        # TODO :
        #  the phase information from the FFT often gets the nature of sin/cosine wrong.
        #  Is there a way to do better?

        avg_y = sum(y_points) / len(y_points)
        zeroed_y_points = [val - avg_y for val in y_points]

        # complex fourier spectrum, with positions adjusted to frequency space
        fft_Ynu = fftshift(fft(zeroed_y_points)) / len(zeroed_y_points)
        fft_nu = fftshift(fftfreq(len(zeroed_y_points), x_points[1] - x_points[0]))

        pos_Ynu = fft_Ynu[ len(fft_Ynu)//2 : ]  # the positive frequency values
        pos_nu = fft_nu[ len(fft_Ynu)//2 : ]    # the positive frequencies

        delta_nu = pos_nu[1] - pos_nu[0]

        # one entry for each positive frequency, either going into sin or cosine
        for n in range( min(len(pos_nu),10) ):
            argmax = np.argmax([np.abs(Ynu) for Ynu in pos_Ynu])

            best_rchisqr = 1e50
            best_freq = None
            is_sin = True

            # try nearby frequencies, explicitly with the data, to see whether sin or cosine is better
            # make this an option in procedural settings
            # note that tested frequencies can overlap from one argmax to the next

            # first with sine
            sin_func = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                         children_list=[PrimitiveFunction.built_in("pow1")])
            test_func1 = CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                           children_list=[PrimitiveFunction.built_in("pow0"),sin_func])
            sin_rchisqr_list = []
            for i in range(7) :
                test_func1.set_args(avg_y, 2*np.abs(pos_Ynu[argmax]), 2*np.pi*(pos_nu[argmax]+delta_nu*(i-3)/3) )
                sin_rchisqr_list.append( self.reduced_chi_squared_of_fit(model=test_func1) )

            # logger(f"{avg_y:.3F}", [ 2*np.pi*(pos_nu[argmax]+delta_nu*(i-3)/3) for i in range(7)])
            # logger(sin_rchisqr_list)

            # should only do this if we find a minimum
            min_idx, min_val = np.argmin(sin_rchisqr_list[1:-1]), min(sin_rchisqr_list[1:-1])  # don't look at endpoints
            if min_val < best_rchisqr :
                list_idx = min_idx + 1
                if min_val < sin_rchisqr_list[list_idx-1] and min_val < sin_rchisqr_list[list_idx+1] :
                    # we found a minimum
                    best_freq = pos_nu[argmax]+delta_nu*(list_idx-3)/3
                    best_rchisqr = min_val
                else :
                    best_freq = pos_nu[argmax]
                    best_rchisqr = min_val

            # now with cosine
            cos_func = CompositeFunction(prim_=PrimitiveFunction.built_in("cos"),
                                          children_list=[PrimitiveFunction.built_in("pow1")])
            test_func2 = CompositeFunction(prim_=PrimitiveFunction.built_in("sum"),
                                           children_list=[PrimitiveFunction.built_in("pow0"),cos_func])
            cos_rchisqr_list = []
            for i in range(7):
                test_func2.set_args( avg_y, 2*np.abs(pos_Ynu[argmax]), 2*np.pi*(pos_nu[argmax] + delta_nu*(i-3) / 3) )
                cos_rchisqr_list.append(self.reduced_chi_squared_of_fit(model=test_func2))

            # should only do this if we find a minimum
            min_idx, min_val = np.argmin(cos_rchisqr_list[1:-1]), min(cos_rchisqr_list[1:-1])  # don't look at endpoints
            if min_val < best_rchisqr:
                is_sin = False
                list_idx = min_idx + 1
                if min_val < cos_rchisqr_list[list_idx - 1] and min_val < cos_rchisqr_list[list_idx + 1]:
                    # we found a minimum
                    best_freq = pos_nu[argmax] + delta_nu*(list_idx-3)/3
                else:
                    best_freq = pos_nu[argmax]
            else :
                pass


            if is_sin :
                # logger(f"Adding {best_freq} to sin")
                self._sin_freq_list.append(best_freq)
            else:
                # logger(f"Adding {best_freq} to cos")
                self._cos_freq_list.append(best_freq)
            pos_Ynu = np.delete(pos_Ynu, argmax)
            pos_nu = np.delete(pos_nu, argmax)

    def find_initial_guess_scaling(self, model):

        scaling_args_no_sign = self.find_set_initial_guess_scaling(model)

        # the above args, with sizes based on scaling, could each be positive or negative. Find the best one (of 2^dof)
        best_rchisqr = 1e50
        best_grid_point = scaling_args_no_sign

        scaling_args_sign_list = []
        # creates list of arguments to try with all +/- sign combinations
        for idx in range( 2**model.dof ) :
            binary_string_of_index = f"0000000000000000{idx:b}"

            new_gridpoint = scaling_args_no_sign.copy()
            for bit, arg in enumerate(new_gridpoint) :
                new_gridpoint[bit] = arg*(1-2*int(binary_string_of_index[-1-bit]))
            scaling_args_sign_list.append(new_gridpoint)

        # tests each of the +/- combinations for the best fit
        # also keep running sums of weigthed points, to try their average at the end
        weighted_point = [ 0 for _ in scaling_args_no_sign ]
        weighted_norm = 0
        for point in scaling_args_sign_list:
            model.set_args( *point )
            temp_rchisqr = self.reduced_chi_squared_of_fit(model)
            weighted_point = list_sums_weights(weighted_point,point,1,1/temp_rchisqr)
            weighted_norm += 1/temp_rchisqr
            if temp_rchisqr < best_rchisqr :
                best_rchisqr = temp_rchisqr
                best_grid_point = point

        # test how good the weighted gridpoint is
        weighted_point_norm = list_sums_weights(weighted_point,[0 for _ in scaling_args_no_sign],1/weighted_norm,1)
        model.set_args(*weighted_point_norm)
        temp_rchisqr = self.reduced_chi_squared_of_fit(model)
        if temp_rchisqr < best_rchisqr and model.num_trig() < 1:  # don't want to mess up frequencies
            best_grid_point = weighted_point_norm
            # best_rchisqr = temp_rchisqr
            # logger(f"Using weighted {best_rchisqr=} {best_grid_point}")

        model.set_args( *best_grid_point )
        # model.print_tree()

        return best_grid_point
    def find_set_initial_guess_scaling(self, composite: CompositeFunction):

        logger(composite, self._sin_freq_list_dup, self._cos_freq_list_dup)

        for child in reversed(composite.children_list) :
            # reverse to bias towards more complicated
            # functions first for fourier frequency setting
            self.find_set_initial_guess_scaling(child)

        # use knowledge of scaling to guess parameter sizes from the characteristic sizes in the data
        # charAvY = (max([datum.val for datum in self._data]) + min([datum.val for datum in self._data])) / 2
        charDiffY = (max([datum.val for datum in self._data]) - min([datum.val for datum in self._data])) / 2
        charAvX = ( max( [datum.pos for datum in self._data] ) + min( [datum.pos for datum in self._data] ) ) / 2
        charDiffX =  (max([datum.pos for datum in self._data]) - min([datum.pos for datum in self._data])) / 2
        if composite.prim.name == "pow0" :
            # this typically represents a shift, so the average X is more important than the range of x-values
            charX = charAvX
        else :
            charX = charDiffX

        # defaults
        xmul = charX**composite.dimension_arg if charX > 0 else charDiffX**composite.dimension_arg
        ymul = 1

        # overrides
        if composite.parent is None :
            if composite.prim.name[:2] != "n_" and composite.prim.name != "sum_" :
                ymul = charDiffY
            else:
                pass
        elif composite.parent.prim.name == "sum_" and composite.parent.parent is None:
            ymul = charDiffY
        elif "cos" in composite.parent.prim.name :
            # in cos( Aexp(Lx) ), for small x the inner composition goes like A + ALx
            # = f(0) + x f'(0) and here f'(0) should correspond to the largest fourier component
            # problem is that this answer should be independent of the initial parameter A... it only works from scratch
            # since we set A = 1 for new composites
            slope_at_zero = (composite.eval_at(2e-5)-composite.eval_at(1e-5) ) / (1e-5 * composite.prim.arg)
            if abs(slope_at_zero) > 1e-5 :
                if len(self._cos_freq_list_dup) > 0 :
                    logger(f"Using cosine frequency {2*np.pi*self._cos_freq_list_dup[0]}")
                    xmul = ( 2*np.pi*self._cos_freq_list_dup.pop(0) ) / slope_at_zero
                else:  # misassigned cosine frequency
                    try:
                        xmul = ( 2*np.pi*self._sin_freq_list_dup.pop(0) ) / slope_at_zero
                    except TypeError :
                        logger("find_set_initial_guess_scaling TypeError",self._sin_freq_list_dup)
                        logger("find_set_initial_guess_scaling TypeError",self._sin_freq_list)
                        logger("find_set_initial_guess_scaling TypeError",self._cos_freq_list_dup)
                        logger("find_set_initial_guess_scaling TypeError",self._cos_freq_list)
                        raise SystemExit
        elif "sin" in composite.parent.prim.name :
            slope_at_zero = (composite.eval_at(2e-5)-composite.eval_at(1e-5) ) / (1e-5 * composite.prim.arg)
            if abs(slope_at_zero) > 1e-5 :
                if len(self._sin_freq_list_dup) > 0 :
                    logger(f"Using sine frequency {2*np.pi*self._sin_freq_list_dup[0]}")
                    xmul = ( 2*np.pi*self._sin_freq_list_dup.pop(0) ) / slope_at_zero
                else:  # misassigned cosine frequency
                    xmul = ( 2*np.pi*self._cos_freq_list_dup.pop(0) ) / slope_at_zero
            logger(f"{xmul=} {self._sin_freq_list} {slope_at_zero*2*np.pi}")
        composite.prim.arg = xmul * ymul

        return composite.get_args()

    def show_fit(self, model=None, pause_on_image = False):

        x_points = []
        y_points = []
        sigma_x_points = []
        sigma_y_points = []

        for datum in self._data :
            x_points.append( datum.pos )
            y_points.append( datum.val )
            sigma_x_points.append( datum.sigma_pos )
            sigma_y_points.append( datum.sigma_val )

        if model is not None:
            plot_model = model.copy()
        else:
            plot_model = self._shown_model



        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
        plt.errorbar( x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        if plot_model is not None :
            smooth_x_for_fit = np.linspace( x_points[0], x_points[-1], 4*len(x_points))
            fit_vals = [ plot_model.eval_at(xi) for xi in smooth_x_for_fit ]
            plt.plot( smooth_x_for_fit, fit_vals, '-', color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        axes = plt.gca()
        if axes.get_xlim()[0] > 0 :
            axes.set_xlim( [0, axes.get_xlim()[1]] )
        if axes.get_ylim()[0] > 0 :
            axes.set_ylim( [0, axes.get_ylim()[1]] )
        axes.set_facecolor( (112/255, 146/255, 190/255) )
        # logger("Optimizer show fit: here once")
        plt.show(block=pause_on_image)

    def inferred_error_bar_size(self) -> float:
        return ( max([datum.val for datum in self._data]) - min([datum.val for datum in self._data]) )/10
        # return np.array([datum.val - model.eval_at(datum.pos) for datum in self._data]).std()
    def chi_squared_of_fit(self,model) -> float:
        if any( [datum.sigma_val < 1e-5 for datum in self._data] ) :
            sumsqr = sum([ (model.eval_at(datum.pos)-datum.val)**2 for datum in self._data ])
            return sumsqr / self.inferred_error_bar_size()**2

        return sum([ (model.eval_at(datum.pos)-datum.val)**2 / datum.sigma_val**2 for datum in self._data ])
    def reduced_chi_squared_of_fit(self,model) -> float:
        k = model.dof
        N = len(self._data)
        return self.chi_squared_of_fit(model) / (N-k) if N > k else 1e5
    def modified_criterion_for_believability(self, model):
        avg_grid_spacing = (self._data[-1].pos - self._data[0].pos) / len(self._data)
        for idx in range(model.num_nodes()):
            node = model.get_node_with_index(idx)
            if node.parent is not None and node.parent.prim.name in ["my_sin", "my_cos"] :
                if node.prim.arg > 10*avg_grid_spacing :
                    logger("LIAR! FREQUENCY EXTRAVAGENT")
                    return self.criterion(model) * (node.prim.arg/avg_grid_spacing)**2
        return self.criterion(model)
    def criterion_w_cov_punish(self, model, cov):
        N = len(model.args)
        cc = coefficient_of_covariance(cov,model.args)
        avg_off = sum_sqr_off_diag(cc) / (N*(N-1)) if N > 1 else 0
        adjustment = avg_off if avg_off > 0.01 else 0
        return self.modified_criterion_for_believability(model) + np.sqrt(adjustment)
    def r_squared(self, model):
        mean = sum( [datum.val for datum in self._data] )/len(self._data)
        variance_data = sum( [ (datum.val-mean)**2 for datum in self._data ] )/len(self._data)
        variance_fit = sum( [ (datum.val-model.eval_at(datum.pos))**2 for datum in self._data ] )/len(self._data)
        return 1 - variance_fit/variance_data


    def Akaike_criterion(self, model):
        # the AIC is equivalent, for normally distributed residuals, to the least chi squared
        k = model.dof
        # N = len(self._data)
        AIC = self.chi_squared_of_fit(model) + 2*k  # we take chi^2 = -2logL + C, discard the C
        return AIC
    def Akaike_criterion_corrected(self, model):
        # correction for small datasets, fixes overfitting
        k = model.dof
        N = len(self._data)
        AICc = self.Akaike_criterion(model) + 2*k*(k + 1) / (N-k-1) if N > k + 1 else 1e5
        return AICc
    def Bayes_criterion(self, model):
        # the same as AIC but penalizes additional parameters more heavily for larger datasets
        k = model.dof
        N = len(self._data)
        BIC = self.chi_squared_of_fit(model) + k*np.log(N)
        return BIC
    def HannanQuinn_criterion(self, model):
        # agrees with AIC at small datasets, but punishes less strongly than Bayes at large N
        k = model.dof
        N = len(self._data)
        HQIC = self.chi_squared_of_fit(model) + 2*k*np.log(np.log(N)) if N > 1 else 1e5
        return HQIC


    def smoothed_data(self, data = None, n=1) -> list[Datum1D]:

        logger("Using smoothed data")

        return_data = []
        if data is None :
            if n <= 1:
                data_to_smooth = [datum.__copy__() for datum in self.average_data()]
            else:
                data_to_smooth = self.smoothed_data(n=n-1)
        else :
            data_to_smooth = data

        for idx, datum in enumerate(data_to_smooth[:-1]):
            new_pos = (data_to_smooth[idx + 1].pos + data_to_smooth[idx].pos) / 2
            new_val = (data_to_smooth[idx + 1].val + data_to_smooth[idx].val) / 2
            # propagation of uncertainty
            new_sigma_pos = np.sqrt((data_to_smooth[idx+1].sigma_pos/2)**2 + (data_to_smooth[idx].sigma_pos/2)**2)
            new_sigma_val = np.sqrt((data_to_smooth[idx+1].sigma_val/2)**2 + (data_to_smooth[idx].sigma_val/2)**2)
            return_data.append(Datum1D(pos=new_pos, val=new_val, sigma_pos=new_sigma_pos, sigma_val=new_sigma_val))

        return return_data
    def deriv_n(self, data, n=1) -> list[Datum1D]:

        # this assumes that data is sequential, i.e. that there are no repeated measurements for each x position
        return_deriv = []
        if n <= 1:
            data_to_diff = data
        else:
            data_to_diff = self.deriv_n(data=data,n=n-1)
        for idx, datum in enumerate(data_to_diff[:-2]):
            new_deriv = ( (data_to_diff[idx+2].val - data_to_diff[idx].val)
                                                   /
                          (data_to_diff[idx+2].pos - data_to_diff[idx].pos)   )
            new_pos   =   (data_to_diff[idx+2].pos + data_to_diff[idx].pos) / 2

            # propagation of uncertainty
            new_sigma_pos = np.sqrt((data_to_diff[idx+2].sigma_pos/2)**2
                                      + (data_to_diff[idx].sigma_pos/2)**2)
            new_sigma_deriv = ((1/(data_to_diff[idx+2].pos - data_to_diff[idx].pos)) *
                               np.sqrt(data_to_diff[idx+2].sigma_val**2 + data_to_diff[idx].sigma_val**2)
                               + new_deriv**2 * (data_to_diff[idx+2].sigma_pos**2 + data_to_diff[idx].sigma_pos**2))
            return_deriv.append(Datum1D(pos=new_pos, val=new_deriv,
                                        sigma_pos=new_sigma_pos, sigma_val=new_sigma_deriv))

        return return_deriv

    # this is only used to find an initial guess for the data, and so more sophisticated techniques like
    # weighted means is not reqired
    def average_data(self) -> list[Datum1D] :

        # get means and number of pos-instances into dict
        sum_val_dict : dict[float,float] = {}
        num_dict : dict[float,int] = {}
        for datum in self._data :
            sum_val_dict[datum.pos] = sum_val_dict.get(datum.pos,0) + datum.val
            num_dict[datum.pos] = num_dict.get(datum.pos,0) + 1

        mean_dict = {}
        for ikey, isum in sum_val_dict.items():
            mean_dict[ikey] = isum / num_dict[ikey]

        propagation_variance_x_dict : dict[float,float] = {}
        propagation_variance_y_dict : dict[float,float] = {}
        sample_variance_y_dict : dict[float,float] = {}
        for datum in self._data:
            # average the variances
            propagation_variance_x_dict[datum.pos] = propagation_variance_x_dict.get(datum.pos,0) + datum.sigma_pos**2
            propagation_variance_y_dict[datum.pos] = propagation_variance_y_dict.get(datum.pos,0) + datum.sigma_val**2
            sample_variance_y_dict[datum.pos] = sample_variance_y_dict.get(datum.pos,0) \
                                                + (datum.val - mean_dict[datum.pos])**2

        averaged_data = []
        for key, val in mean_dict.items():
            sample_uncertainty_squared = sample_variance_y_dict[key] / (num_dict[key] - 1) if num_dict[
                                                                                                  key] > 1 else 0
            propagation_uncertainty_squared = propagation_variance_y_dict[key] / num_dict[key]
            ratio = (sample_uncertainty_squared / (sample_uncertainty_squared + propagation_uncertainty_squared)
                     if propagation_uncertainty_squared > 0 else 1)

            # interpolates smoothly between 0 uncertainty in data points (so all uncertainty comes from sample spread)
            # to the usual uncertainty coming from both the data and the spread
            # TODO: see
            #  https://stats.stackexchange.com/questions/454120/how-can-i-calculate-uncertainty-of-the-mean-of-a-set-of-samples-with-different-u
            # for a more rigorous treatment
            effective_uncertainty_squared = ratio * sample_uncertainty_squared + (
                        1 - ratio) * propagation_uncertainty_squared

            averaged_data.append(Datum1D(pos=key, val=val,
                                         sigma_pos=np.sqrt(propagation_variance_x_dict[key]),
                                         sigma_val=np.sqrt(effective_uncertainty_squared)
                                         )
                                 )
        return sorted(averaged_data)

    def load_default_functions(self):
        self._primitive_function_list.extend( [ PrimitiveFunction.built_in("pow0"),
                                                PrimitiveFunction.built_in("pow1") ] )
    def load_non_defaults_to_primitive_function_list(self) :
        for key, use_function in self._use_functions_dict.items():
            if key == "cos(x)" and use_function:
                self._primitive_function_list.append( PrimitiveFunction.built_in("cos") )
            if key == "sin(x)" and use_function:
                self._primitive_function_list.append( PrimitiveFunction.built_in("sin") )
            if key == "exp(x)" and use_function:
                self._primitive_function_list.append( PrimitiveFunction.built_in("exp") )
            if key == "log(x)" and use_function:
                self._primitive_function_list.append( PrimitiveFunction.built_in("log" ) )
            if key == "1/x" and use_function:
                self._primitive_function_list.append(PrimitiveFunction.built_in("pow_neg1"))
            # if key == "x\U000000B2" and use_function:
            #     self._primitive_function_list.append(PrimitiveFunction.built_in("pow2"))
            # if key == "x\U000000B3" and use_function:
            #     self._primitive_function_list.append(PrimitiveFunction.built_in("pow3"))
            # if key == "x\U00002074" and use_function:
            #     self._primitive_function_list.append(PrimitiveFunction.built_in("pow4"))
            # custom functions are also loaded, but elsewhere, using add_primitive_to_list()


def std(cov):
    return list(np.sqrt(np.diagonal(cov)))
def correlation_from_covariance(cov_mat: np.ndarray) -> np.ndarray:
    v = np.sqrt(np.diag(cov_mat))
    outer_v = np.outer(v, v)
    correlation = cov_mat / outer_v
    correlation[cov_mat == 0] = 0
    return correlation
def coefficient_of_covariance(cov_mat: np.ndarray, mean_list: list[float]) -> np.ndarray :
    cc = cov_mat.copy()
    for idx, (cov_i, mean_i) in enumerate( zip(cov_mat, mean_list) ) :
        for jdx, (cov_ij, mean_j) in enumerate(zip(cov_i, mean_list) ):
            cc[idx][jdx] = cov_ij/(mean_i*mean_j)
    return cc

def sum_sqr_off_diag(cc: np.ndarray) -> float :
    sumsqr = 0
    for (idx, jdx), cc_ij in np.ndenumerate(cc):
        if idx < jdx :
            sumsqr += cc_ij**2
    return sumsqr
def list_sums_weights(l1,l2,w1,w2) -> list[float]:
    if len(l1) != len(l2) :
        return []
    sum_list = []
    for val1, val2 in zip(l1,l2) :
        sum_list.append(val1*w1+val2*w2)
    return sum_list
