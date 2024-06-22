"""Implements the Optimizer class, which holds data and keeps track of the possible functional
models that may used to fit that data. After fitting the data to many model, the Optimizer
also tracks the top5-ranked models."""

# default libraries
import random
import re as regex
from typing import Callable, Iterator, Optional
from copy import copy

# https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64
from io import BytesIO
from base64 import b64encode

# external libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# pylint has a hard time with fft, see
# https://stackoverflow.com/questions/65880880/no-name-fft-in-module-scipy
from scipy.fft import fft, fftshift, fftfreq  # pylint: disable=no-name-in-module

# following pylint-disables imports needed for runtime `exec` usage
import scipy.stats  # pylint: disable=unused-import
import scipy.special  # pylint: disable=unused-import

# internal classes
from autofit.src.datum1D import Datum1D
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.composite_function import CompositeFunction
from autofit.src.package import logger
from autofit.src.algorithms import find_window_left, find_window_right, bisect
import autofit.src.plot_formatter as pf


class Optimizer:
    """
    Finds the top5 models for the current dataset. When a new dataset is provided,
    it can either generate a new set of top5 models, or evaluate those top5 models on
    the new dataset. The list of potential models is generated here from compositions/sums/products
     of primitive functions, with a host of rules to trim the list down to a manageable number.

    The models are ranked on a particular criterion (reduced chi-squared/AIC/BIC/HQIC).
    Fitting is done using scipy.optimize.curve-fit. Initial parameters are determined
    through a combination of scaling arguments, Fourier series coefficients,
    and smoothed peak finding.
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        data: list[Datum1D] = None,
        use_functions_dict: dict[str, bool] = None,
        max_functions: int = 3,
        regen: bool = True,
        criterion: str = "rchisqr",
    ):

        logger("New optimizer created")

        # datasets, which are lists of Datum1D instances
        self._data: list[Datum1D] = []
        # I.e., the raw datapoints of (x,y), possibly with uncertainties. It may
        # also represent a binned point if the input file is a list of x_values only

        # fit results
        self._shown_model: CompositeFunction = None  # a CompositeFunction
        self._shown_covariance: Optional[np.ndarray] = None
        self._shown_rchisqr: float = 1e100

        # top 5 models and their data
        self._top5_models: list[CompositeFunction] = []
        self._top5_covariances: list[np.ndarray] = []
        self._top5_rchisqrs: list[float] = [1e100]  # [1e5  for _ in range(5)]

        # function construction parameters
        self._max_functions: int = max_functions
        self._primitive_function_list: list[PrimitiveFunction] = []
        self._primitive_names_list: list[str] = []

        # useful auxiliary varibles
        self._temp_function: Optional[CompositeFunction] = None  # for ?
        self._cos_freq_list: list[float] = []
        self._sin_freq_list: list[float] = []
        self._cos_freq_list_dup: list[float] = []
        self._sin_freq_list_dup: list[float] = []

        if data is not None:
            # list of Datum1D. Sort makes the data monotone increasing in x-value
            self._data = sorted(data)

        if use_functions_dict is None:
            use_functions_dict = {}
        self._use_functions_dict: dict[str, bool] = use_functions_dict
        self.load_default_functions()
        self.load_non_defaults_to_primitive_function_list()

        self._composite_function_list = []
        self._composite_generator: Optional[Iterator[CompositeFunction]] = None
        self._gen_idx = -1

        self._criterion: Callable[[CompositeFunction], float] = self.reduced_chi_squared_of_fit
        if criterion == "AIC":
            self._criterion = self.akaike_criterion
        elif criterion == "AICc":
            self._criterion = self.akaike_criterion_corrected
        elif criterion == "BIC":
            self._criterion = self.bayes_criterion
        elif criterion == "HQIC":
            self._criterion = self.hannan_quinn_criterion

        self._regen_composite_flag = regen
        self._try_harder = True

    def __repr__(self):
        return f"Optimizer with {self._max_functions} m.f."

    @property
    def shown_model(self) -> CompositeFunction:
        return self._shown_model

    @shown_model.setter
    def shown_model(self, other: CompositeFunction) -> None:
        self._shown_model = other

    @property
    def shown_parameters(self) -> list[float]:
        return self._shown_model.args

    @shown_parameters.setter
    def shown_parameters(self, args_list: list[float]) -> None:
        self._shown_model.args = args_list

    @property
    def shown_uncertainties(self) -> list[float]:
        if self._shown_covariance is None:
            raise RuntimeError
        return list(np.sqrt(np.diagonal(self._shown_covariance)))

    @property
    def shown_covariance(self) -> np.ndarray:
        return self._shown_covariance

    @shown_covariance.setter
    def shown_covariance(self, cov_np: np.ndarray):
        self._shown_covariance = cov_np

    @property
    def shown_rchisqr(self) -> float:
        return self._shown_rchisqr

    @shown_rchisqr.setter
    def shown_rchisqr(self, val: float) -> None:
        self._shown_rchisqr = val

    @property
    def shown_cc(self) -> np.ndarray:
        """The coefficient of covariance, i.e. cc_ij = cov_ij / mu_i mu_j"""
        return coefficient_of_covariance(self.shown_covariance, self.shown_parameters)

    @property
    def shown_cc_offdiag_sumsqrs(self) -> float:
        return sum_sqr_off_diag(self.shown_cc)

    @property
    def avg_off(self) -> float:
        """Average of the off-diagonal sum of squares of the coefficient of covariance"""
        N = len(self.shown_parameters)
        return self.shown_cc_offdiag_sumsqrs / (N * (N - 1)) if N > 1 else 0

    @property
    def top5_models(self) -> list[CompositeFunction]:
        """THe top 5 models that have been fitted to the current dataset"""
        return self._top5_models

    @top5_models.setter
    def top5_models(self, other):
        """Manually sets the top 5 models"""
        self._top5_models = other

    @property
    def top_model(self) -> Optional[CompositeFunction]:
        if len(self.top5_models) > 0:
            return self.top5_models[0]
        logger("Optimier.top_model(): No models in the top 5")
        return None

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

    def top5_maccs(self):
        top5_mean_absolute_ccs = []
        for cov, model in zip(self.top5_covariances, self.top5_models):
            top5_mean_absolute_ccs.append(self._mean_abs_coc(model, cov))
        return top5_mean_absolute_ccs

    @property
    def top_rchisqr(self):
        return self.top5_rchisqrs[0]

    @property
    def prim_list(self) -> list[PrimitiveFunction]:
        return self._primitive_function_list

    def remove_named_from_prim_list(self, name: str):
        for prim in self.prim_list[:]:
            if prim.name == name:
                self._primitive_function_list.remove(prim)
                return

        logger(f"No prim named {name} in optimizer prim list")

    @property
    def criterion(self) -> Callable[[CompositeFunction], float]:
        return self._criterion

    @criterion.setter
    def criterion(self, other: Callable[[CompositeFunction], float]):
        self._criterion = other
        self.update_top5_rchisqrs_for_new_data(self._data)

    @property
    def composite_function_list(self) -> list[CompositeFunction]:
        """
        Returns the list of models the optimizer will currently test
        """
        return self._composite_function_list

    @composite_function_list.setter
    def composite_function_list(self, comp_list) -> None:
        self._composite_function_list = comp_list

    @property
    def gen_idx(self) -> int:
        return self._gen_idx

    def update_opts(self, use_functions_dict: dict[str, bool], max_functions: int):

        self._use_functions_dict = use_functions_dict
        self._max_functions = max_functions

        self._primitive_function_list = []
        self.load_default_functions()
        self.load_non_defaults_to_primitive_function_list()

        self._regen_composite_flag = True

    def update_top5_rchisqrs_for_new_data(self, new_data):
        logger(
            f"Optimizer.update_top5_rchisqrs_for_new_data(): "
            f"Updating top5 criterions. Before: {self.top5_rchisqrs}"
        )
        self.set_data_to(new_data)
        for idx, model in enumerate(self.top5_models):
            self.top5_rchisqrs[idx] = self.criterion(model)
        logger(f"Optimizer.update_top5_rchisqrs_for_new_data(): After: {self.top5_rchisqrs}")

    # changes top5 lists, does not change _shown variables
    def query_add_to_top5(self, model: CompositeFunction, covariance):

        rchisqr = self.criterion(model)
        logger(f"Querying {rchisqr:.2F} for {model.name}")
        if (
            np.isnan(rchisqr)
            or rchisqr > self.top5_rchisqrs[-1]
            or any(V < 0 for V in np.diagonal(covariance))
        ):
            return

        rchisqr_adjusted = self._criterion_modified(model, covariance)
        if rchisqr_adjusted > rchisqr:
            logger(f"Adjustment: {rchisqr} -> {rchisqr_adjusted}")
            rchisqr = rchisqr_adjusted
        # check for duplication
        for idx, (topper, chisqr) in enumerate(zip(self.top5_models[:], self.top5_rchisqrs[:])):
            # comparing with names
            if model.longname == topper.longname:
                logger("Same name in query")
                if chisqr <= rchisqr + 1e-5:
                    return
                # delete the original entry from the list because ???
                del self.top5_models[idx]
                del self.top5_covariances[idx]
                del self.top5_rchisqrs[idx]
                break
            if abs(rchisqr - chisqr) / (rchisqr + chisqr) < 1e-5:
                # essentially the same function, need to choose which
                # one is the better representative

                # dof should trump all
                if topper.dof < model.dof:
                    logger(
                        f"Booting out contender {model.name} with dof {model.dof} "
                        f"in favour of {topper.name} with dof {topper.dof}"
                    )
                    return
                if topper.dof > model.dof:
                    logger(
                        f"Keeping contender {model.name} with dof {model.dof} "
                        f"but also keeping {topper.name} with dof {topper.dof}"
                    )
                    break
                # depth should then be discouraged? TODO: check this
                if topper.depth < model.depth:
                    logger(
                        f"Booting out contender {model.name} with depth {model.depth} "
                        f"in favour of {topper.name} with depth {topper.depth}"
                    )
                    return
                # width should then be minimized
                if topper.width < model.width:
                    logger(
                        f"Booting out contender {model.name} with width {model.width} "
                        f"in favour of {topper.name} with width {topper.width}"
                    )
                    return
                # check the coefficients of covariance
                cc_topper = self._mean_abs_coc(self._top5_models[idx], self._top5_covariances[idx])
                cc_model = self._mean_abs_coc(model, covariance)
                if cc_topper < cc_model:
                    logger(
                        f"Booting out contender {model.name} with CoC {cc_model}, "
                        f"in favour of {topper.name} with CoC {cc_topper}"
                    )
                    # default is to keep the first one added
                    return
                if (
                    topper.dof == model.dof
                    and topper.depth == model.depth
                    and topper.width == model.width
                ):
                    logger(
                        f"Booting out contender {model.name} in favour of {topper.name}, "
                        f"both with with dof, depth, width "
                        f"{topper.dof} {topper.depth} {topper.width}"
                    )
                    # default is to keep the first one added
                    return
                # more sophisticated distinguishers might also try to
                # minimize correlations between parameters
                logger(f"Booting out {topper.name} in favour of contender {model.name}")
                del self.top5_models[idx]
                del self.top5_covariances[idx]
                del self.top5_rchisqrs[idx]
                break
                # after dof, depth should trump all

        for idx, _ in enumerate(self._top5_rchisqrs[:]):
            if rchisqr < self._top5_rchisqrs[idx]:
                self.top5_models.insert(idx, model.copy())
                self.top5_covariances.insert(idx, covariance)
                self.top5_rchisqrs.insert(idx, rchisqr)

                self.top5_models = self._top5_models[:5]
                self.top5_covariances = self._top5_covariances[:5]
                self.top5_rchisqrs = self._top5_rchisqrs[:5]

                par_str_list = [f"{arg:.3F}" for arg in model.args]
                unc_str_list = [f"{unc:.3F}" for unc in std(covariance)]

                logger(
                    f"New top {idx} with red_chisqr={rchisqr:.2F}: "
                    f"{model=} with pars=["
                    + ", ".join(par_str_list)
                    + "] \u00B1 ["
                    + ", ".join(unc_str_list)
                    + "]"
                )
                model.print_tree()

                # also check any parameters for being "equivalent" to zero. If so, remove the d.o.f.
                # and add the new function to the top5 list
                for ndx, (arg, unc) in enumerate(zip(model.args, std(covariance))):
                    if (
                        not np.isinf(unc) and abs(arg) < 2 * unc
                    ):  # 95% confidence level arg is not different from zero

                        reduced_model = model.submodel_without_node_idx(ndx)
                        if (
                            reduced_model is None
                            or self.fails_rules(reduced_model)
                            or (
                                reduced_model.prim.name == "sum_"
                                and reduced_model.num_children() == 0
                            )
                        ):
                            logger("Zero arg detected but but can't reduce the model")
                            continue
                        logger("Zero arg detected: new trimmed model is")
                        reduced_model.set_submodel_of_zero_idx(model, ndx)
                        reduced_model.print_tree()
                        if reduced_model.name in self.top5_names:
                            reduced_idx = self.top5_names.index(reduced_model.name)
                            if reduced_idx < idx:
                                logger("Optimizer.query_add(): Removing super-model")
                                # don't keep the supermodel
                                del self.top5_models[idx]
                                del self.top5_covariances[idx]
                                del self.top5_rchisqrs[idx]
                                break
                        # this method changes shown models
                        improved_reduced, improved_cov = self.fit_this_and_get_model_and_covariance(
                            model_=reduced_model,
                            initial_guess=reduced_model.args,
                            change_shown=False,
                        )
                        self.query_add_to_top5(model=improved_reduced, covariance=improved_cov)
                        break
                return

    def composite_function_generator(
        self, depth, regen_built_ins=True
    ) -> Iterator[CompositeFunction]:

        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        if depth == 0:
            if regen_built_ins:
                logger("Composite generator:", self._primitive_names_list)
                logger("Starting new generator at 0 depth")
                self._gen_idx = 0
                yield from CompositeFunction.built_in_list()

            for iprim in self._primitive_function_list:
                new_comp = CompositeFunction(
                    prim_=PrimitiveFunction.built_in("sum"), children_list=[iprim]
                )
                yield new_comp

        else:
            head_gen = self.composite_function_generator(depth=depth - 1, regen_built_ins=False)
            for icomp in head_gen:
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list:

                        # sums
                        new_comp = icomp.copy()
                        sum_node = new_comp.get_node_with_index(idescendent)
                        if (
                            sum_node.num_children() > 0
                            and iprim.name > sum_node.children_list[-1].prim.name
                        ):
                            pass
                        else:
                            if sum_node.prim.name in [
                                "my_sin",
                                "my_cos",
                            ] and iprim.name in ["my_sin", "my_cos"]:
                                pass  # speedup for double trig
                            elif sum_node.prim.name == "my_log" and iprim.name in [
                                "my_log",
                                "my_exp",
                            ]:
                                pass
                            elif sum_node.prim.name == "my_exp" and iprim.name in [
                                "my_exp",
                                "pow0",
                            ]:
                                pass
                            else:
                                sum_node.add_child(iprim, update_name=True)
                                yield new_comp

                        # factors
                        new_mul = icomp.copy()
                        mul_node = new_mul.get_node_with_index(idescendent)
                        if mul_node.prim.name >= iprim.name:
                            if mul_node.prim.name == "my_exp" and iprim.name in [
                                "my_exp",
                                "pow0",
                            ]:
                                pass  # speedup for multiplied exps
                            else:
                                mul_node.add_younger_brother(iprim, update_name=True)
                                yield new_mul

    def valid_composite_function_generator(self, depth) -> Iterator[CompositeFunction]:

        all_comps_at_depth = self.composite_function_generator(depth)
        for icomp in all_comps_at_depth:
            if not self.fails_rules(icomp):
                yield icomp

    def all_valid_composites_generator(self) -> Iterator[CompositeFunction]:
        for idepth in range(7):
            for icomp in self.valid_composite_function_generator(depth=idepth):
                self._gen_idx += 1
                yield icomp

    # TODO: make a generator version of this e.g. [] -> ()
    def build_composite_function_list(self, status_bar=None):  # status_bar : tk_label
        # the benefit of using this is that you can generate it once,
        # and if the options don't change you don't need
        # to generate it again. Follow that logic
        if not self._regen_composite_flag:
            return
        logger(
            f"{self._regen_composite_flag}, so regenerating composite "
            f"list with {self._max_functions} "
            f"and {[prim.name for prim in self._primitive_function_list]}"
        )
        self._regen_composite_flag = False

        # start with simple primitives in a sum
        self._composite_function_list = []
        for iprim in self._primitive_function_list:
            new_comp = CompositeFunction(
                prim_=PrimitiveFunction.built_in("sum"), children_list=[iprim]
            )
            self._composite_function_list.append(new_comp)

        last_list = self._composite_function_list
        for depth in range(self._max_functions - 1):
            new_list: list[CompositeFunction] = []
            for icomp in last_list:
                if status_bar is not None:
                    status_bar.configure(
                        text=f"   Stage 1/3: {len(last_list)+len(new_list):>10} "
                        f"naive models generated, {0:>10} models fit."
                    )
                    status_bar.master.master.update()
                    if status_bar["bg"] == "#010101":  # cancel code
                        self._regen_composite_flag = True
                        break
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list[:]:

                        new_comp = icomp.copy()
                        sum_node = new_comp.get_node_with_index(idescendent)
                        # the naming already sorts the multiplication and
                        # summing parts by descending order,
                        # so why not just build that in
                        if (
                            sum_node.num_children() > 0
                            and iprim.name > sum_node.children_list[-1].prim.name
                        ):
                            pass
                        else:
                            if sum_node.prim.name in [
                                "my_sin",
                                "my_cos",
                            ] and iprim.name in ["my_sin", "my_cos"]:
                                pass  # speedup for double trig
                            elif sum_node.prim.name == "my_log" and iprim.name in [
                                "my_log",
                                "my_exp",
                            ]:
                                pass
                            elif sum_node.prim.name == "my_exp" and iprim.name in [
                                "my_exp",
                                "pow0",
                            ]:
                                pass
                            else:
                                sum_node.add_child(iprim, update_name=True)
                                new_list.append(new_comp)

                        new_mul = icomp.copy()
                        mul_node = new_mul.get_node_with_index(idescendent)
                        if mul_node.prim.name >= iprim.name:
                            if mul_node.prim.name == "my_exp" and iprim.name in [
                                "my_exp",
                                "pow0",
                            ]:
                                pass  # speedup for multiplied exps
                            else:
                                mul_node.add_younger_brother(iprim, update_name=True)
                                new_list.append(new_mul)
                if status_bar is not None:
                    if status_bar["bg"] == "#010101":  # cancel code
                        break
            logger(f"{depth} build_comp_list new_len=", len(new_list))
            self._composite_function_list.extend(new_list)
            last_list = new_list

        for comp in CompositeFunction.built_in_list():
            self._composite_function_list.append(comp.copy())

        # prepend the current top 5 models

        self.trim_composite_function_list(status_bar=status_bar)
        logger(f"After trimming list: (len={len(self._composite_function_list)})")
        for icomp in self._composite_function_list:
            logger(icomp)
        logger("|----------------\n")

    def trim_composite_function_list(self, status_bar):
        # Finds and removes duplicates
        # Performs basic algebra to recognize simpler forms with fewer parameters
        # E.g. Functions like pow0(pow0) and pow1(pow1 + pow1) have fewer
        # arguments than the naive calculation expects

        # Only run this at the end of the tree generation: if you run this in intermediate steps,
        # you will miss out on some functions possibilities

        num_comps = len(self._composite_function_list[:])
        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        # use regex to trim based on rules applied to composite names
        for idx, icomp in enumerate(self._composite_function_list[:]):

            if idx % 500 == 0:
                if status_bar is not None:
                    status_bar.configure(
                        text=f"   Stage 2/3: {len(self._composite_function_list):>10} valid "
                        f"models generated, {0:>10} models fit."
                    )
                    status_bar.master.master.update()
                    if status_bar["bg"] == "#010101":  # cancel code
                        self._regen_composite_flag = True
                        break
                logger(f"{idx}/{num_comps}")

            # if self.fails_rules(icomp) :
            if self.validate_fails(icomp):
                self._composite_function_list.remove(icomp)

        # remove duplicates using a dict {}
        self._composite_function_list = list(
            {repr(icomp): icomp for icomp in self._composite_function_list[:]}.values()
        )

    def fails_rules(self, icomp):

        name = icomp.name

        """
        I imagine non-regex is faster than regex
        """

        if (
            icomp.prim.name == "sum_"
            and icomp.num_children() < 2
            and icomp.younger_brother is not None
        ):
            return 1
        # trig inside trig is too complex, and very (very) rarely occurs in applications
        if icomp.has_double_trigness() or icomp.has_double_expness() or icomp.has_double_logness():
            return 29
        # sins, cosines, exps, and logs never have angular frequency or decay parameters exactly 1
        # all unitless-argument functions start with my_
        if icomp.has_argless_explike():
            return 43
        # pow1 with composition of exactly one term
        if icomp.has_trivial_pow1():
            return 13 * 17
        # pow1 used as a sub-sum inside head sum
        for child in icomp.children_list:
            if (
                child.prim.name == "pow1"
                and child.num_children() > 0
                and child.younger_brother is None
                and child.older_brother is None
            ):
                return 97

        # repeated reciprocal is wrong
        if icomp.has_repeated_reciprocal():
            return 7
        # pow1 times pow1,2,3,4 is wrong
        if icomp.has_reciprocal_cancel_pospow():
            return 61
        if icomp.has_log_with_odd_power():
            return 67
        if icomp.has_pow_with_no_sum():
            return 5
        if (
            icomp.prim.name == "my_log"
            and icomp.num_children() > 0
            and icomp.children_list[0].prim.name in ["my_sin", "my_cos"]
        ):
            return 113

        # composition of a constant function is wrong
        if regex.search(r"pow0\(", name):
            return 2
        if regex.search(r"\(pow0\)", name):
            return 3
        # composition of powers with no sum is wrong -- not strictly true -- think of log^2(Ax+B)
        if regex.search(r"pow[0-9]\(pow[a-z0-9_·]*\)", name):
            return 5
        # trivial reciprocal is wrong
        if regex.search(r"pow_neg1\(pow1\)", name):
            return 11

        # sum of the same function (without further composition) is wrong
        for prim_name in self._primitive_names_list:
            if regex.search(f"{prim_name}[a-z0-9_+]*{prim_name}", name):
                return 23

        # pow0+log(...) is a duplicate since A + Blog( Cf(x) ) = B log( exp(A/B) Cf(x) ) = log(f)
        if regex.search(r"pow0\+my_log\([a-z0-9_]*\)", name) or regex.search(
            r"my_log\([a-z0-9_]*\)\+pow0", name
        ):
            return 37

        # pow3(exp(...)) is just exp(3...)
        if regex.search(r"pow[a-z0-9_]*\(my_exp\([a-z0-9_+]*\)\)", name):
            return 83

        """
        Multiplicative rules
        """
        if regex.search("pow0·", name) or regex.search("·pow0", name):
            return 1000 + 2

        return 0

    def validate_fails(self, icomp):

        name = icomp.name
        remove_flag = self.fails_rules(icomp)

        # second check on the fails we did or didnt experience
        good_list = [
            "my_exp(my_log)",
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
            "my_sin(pow0+my_exp(pow1))",
            "pow4+pow1",
        ]

        for good in good_list:
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

        if name in ["pow1·my_exp(my_exp)", "pow1·my_exp(pow1·my_exp)"] and not remove_flag:
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=29 <<<\n\n")
            logger(icomp.has_double_expness())
            raise SystemExit

        if name == "my_exp(pow1)·my_exp(pow1)":
            logger(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=?? <<<\n\n")
            raise RuntimeWarning

        return remove_flag

    def add_primitive_to_list(self, name: str, functional_form: str) -> str:
        """For adding one's own Primitive Functions to the built-in list"""

        if self._use_functions_dict["custom"] != 1:
            return ""
        if name in [prim.name for prim in self._primitive_function_list]:
            return ""

        # logger(f"\n{name} {functional_form}")
        if (
            regex.search("\\\\", functional_form)
            or regex.search("\n", functional_form)
            or regex.search(r"\s", functional_form)
        ):
            return "Stop trying to inject code"

        # search for autofit-specific strings inside the functional form
        # this allows chaining of user-defined primitives
        for key in PrimitiveFunction.built_in_dict():
            # raise NotImplementedError
            pattern = f"{key}" + r"\("
            functional_form_regexed = regex.sub(
                pattern,
                f"""PrimitiveFunction.built_in("{key}").eval_at(""",
                functional_form,
            )
            if functional_form_regexed != functional_form:
                functional_form = functional_form_regexed

        code_str = f"def {name}(x,arg):\n"
        code_str += f"    return arg*({functional_form})\n"
        code_str += f"new_prim = PrimitiveFunction(func={name})\n"
        code_str += "dict = PrimitiveFunction.built_in_dict()\n"
        code_str += f'dict["{name}"] = new_prim\n'

        try:
            exec(code_str)
        except SyntaxError:
            return (
                f"Corrupted custom function {name} "
                f"with form={functional_form}, returning to blank slate."
            )

        # logger("Optimizer.add_primitive_to(): listing built-in dict items...")
        # for key, val in PrimitiveFunction.built_in_dict().items() :
        #     logger(f"Key: {key} Val: {val}")

        try:
            PrimitiveFunction.built_in(name).eval_at(np.pi / 4)
        except NameError:
            return (
                f"One of the functions used in your custom function {name} \n"
                f"    with form {functional_form} does not exist."
            )

        self._primitive_function_list.append(PrimitiveFunction.built_in(name))

        return ""

    def set_data_to(self, other_data):
        self._data = sorted(other_data)

    # does not change self._data
    def fit_setup(
        self, fourier_condition=True
    ) -> tuple[list[float], list[float], list[float], bool]:

        x_points = []
        y_points = []
        sigma_points = []

        use_errors = True
        y_range = +max((datum.val for datum in self._data)) - min(
            (datum.val for datum in self._data)
        )
        for datum in self._data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            if datum.sigma_val < 1e-10:
                use_errors = False
                sigma_points.append(y_range / 10)
                # datum.sigma_val = y_range/10
                # if you want chi_sqr() to work for zero-error data, you need the above instead
            else:
                sigma_points.append(datum.sigma_val)

        # do an FFT if there's a sin/cosine --
        # we zero-out the average height of the data to remove
        # the 0-frequency mode as a contribution
        # then to pass in the dominant frequencies as arguments to the initial guess
        if fourier_condition:
            self.create_cos_sin_frequency_lists()

        return x_points, y_points, sigma_points, use_errors

    # does not change input model_
    # changes _shown variables
    # does not change top5 lists
    def fit_loop(
        self,
        model_,
        x_points,
        y_points,
        sigma_points,
        use_errors,
        initial_guess=None,
        info_string="",
    ):

        model = model_.copy()

        if model.num_trig() > 0 and initial_guess is None:
            self._cos_freq_list_dup = self._cos_freq_list.copy()
            self._sin_freq_list_dup = self._sin_freq_list.copy()

        logger(f"\nFitting {model=}")
        logger(model.tree_as_string_with_dimensions())

        # Find an initial guess for the parameters based off scaling arguments
        if initial_guess is None:
            if "Pow" not in model.name:
                initial_guess = self.find_initial_guess_scaling(model)
            else:
                degree = model.name.count("+")
                np_args = np.polyfit(x_points, y_points, degree)
                leading = np_args[0]
                trailing = np_args[1:] / leading
                initial_guess = [leading] + list(trailing)

        logger(f"{info_string}Scaling guess: {initial_guess}")

        # Next, find a better guess by relaxing the error bars on the data
        # Unintuitively, this helps. Tight error bars flatten the gradients
        # away from the global minimum, and so relaxed error bars help point towards global minima
        try:
            better_guess, better_cov = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                model.scipy_func_smoothed,
                xdata=x_points,
                ydata=y_points,
                p0=initial_guess,
                maxfev=500,
                method="lm",
            )
            if any((x < 0 for x in np.diagonal(better_cov))):
                logger("Negative variance encountered")
                raise RuntimeError
        except RuntimeError:
            logger("Couldn't find optimal parameters for better guess.")
            model.args = list(initial_guess)
            return model, np.array([1e10 for _ in range(len(initial_guess) ** 2)]).reshape(
                len(initial_guess), len(initial_guess)
            )
        except TypeError:
            logger("Too many dof for dataset.")
            model.args = list(initial_guess)
            return model, np.array([1e10 for _ in range(len(initial_guess) ** 2)]).reshape(
                len(initial_guess), len(initial_guess)
            )
        logger(f"{info_string}Better guess: {better_guess} +- {np.sqrt(np.diagonal(better_cov))}")

        # TODO: delete this graient descent if you can't get it working
        if model.name == "adsawpow_neg1+pow1":
            # and any(np.isinf(np.sqrt(np.diagonal(better_cov)))) :
            # Now do manual "gradient" descent for a bit. Seems to get stuck on trivial tasks
            for idx in range(model.dof):
                curr_arg = model.get_arg_i(idx)
                tmp_model = model.copy()
                lower = self.chi_squared_of_fit(tmp_model)
                if lower > 1000:
                    break
                best = lower
                grad = 1e5
                decay = 0
                while abs(grad) > 1e-2:
                    tmp_model.set_arg_i(idx, curr_arg + 1e-5)
                    upper = self.chi_squared_of_fit(tmp_model)
                    grad = 0.9**decay * (upper - lower) / 1e-5
                    print(model.name, idx, upper, lower, grad)
                    curr_arg -= 0.1 * grad  # update
                    tmp_model.set_arg_i(idx, curr_arg)
                    lower = self.chi_squared_of_fit(tmp_model)
                    if lower >= best:
                        # update failed
                        print(f"Update failed on idx {idx}")
                        break
                    decay += 1

                model.set_arg_i(idx, curr_arg + 0.1 * grad)

        # Finally, use the better guess to find the true minimum with the true error bars
        try:
            np_pars, np_cov = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                model.scipy_func,
                xdata=x_points,
                ydata=y_points,
                sigma=sigma_points,
                absolute_sigma=use_errors,
                p0=better_guess,
                maxfev=1000,
            )
            if any((x < 0 for x in np.diagonal(np_cov))):
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
        if any((datum.sigma_pos**2 / (datum.pos**2 + 1e-10) > 1e-10 for datum in self._data)):
            for it in range(3):  # iterate 3 times
                effective_sigma = np.sqrt(
                    [
                        datum.sigma_val**2
                        + datum.sigma_pos**2 * model.eval_deriv_at(datum.pos) ** 2
                        for datum in self._data
                    ]
                )
                try:
                    np_pars, np_cov = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                        model.scipy_func,
                        xdata=x_points,
                        ydata=y_points,
                        sigma=effective_sigma,
                        absolute_sigma=use_errors,
                        p0=np_pars,
                        maxfev=5000,
                    )
                except RuntimeError:
                    logger(f"On model {model} max_fev reached")
                    # raise RuntimeError
                model.args = list(np_pars)
                logger(
                    f"Now after {it+1} refits, with effective variance: "
                    f"{np_pars} +- {np.sqrt(np.diagonal(np_cov))}"
                )

        return model, np_cov

    # does not change input model_
    # changes _shown variables, unless toggled off
    # does not change top5 lists
    def fit_this_and_get_model_and_covariance(
        self,
        model_: CompositeFunction,
        initial_guess=None,
        change_shown=True,
        do_halving=False,
        halved=0,
    ):

        model = model_.copy()
        init_guess = initial_guess
        if init_guess is None:
            while model.is_submodel:
                supermodel = model_.submodel_of.copy()
                logger(
                    f"\n>>> {model_}: Have to go back to the supermodel {supermodel} "
                    f"for refitting. {model_.submodel_zero_index}"
                )
                supermodel, _ = self.fit_this_and_get_model_and_covariance(
                    model_=supermodel,
                    change_shown=False,
                    do_halving=do_halving,
                    halved=halved,
                )
                logger(
                    f"rchisqr for supermodel= {self.reduced_chi_squared_of_fit(supermodel)}"
                    f" {do_halving=} {halved=}"
                )
                model = supermodel.submodel_without_node_idx(model_.submodel_zero_index)
                # ^ creates a submodel that doesnt think it's a submodel
                init_guess = model.args
        if model.name[-8:] == "Gaussian":
            modal = model.num_children()
            if modal > 1:
                means = self.find_peaks_for_gaussian(expected_n=modal)
                logger(f"Peaks expected at {means}")
                widths = self.find_widths_for_gaussian(means=means)
                logger(f"Widths expected to be {widths}")
                est_amplitudes = self.find_amplitudes_for_gaussian(means=means, widths=widths)
                logger(f"Amplitudes expected to be {est_amplitudes}")
                init_guess = []
                for amp, width, mean in zip(est_amplitudes, widths, means):
                    init_guess.extend([amp, width, mean])

        x_points, y_points, sigma_points, use_errors = self.fit_setup()

        fitted_model, fitted_cov = self.fit_loop(
            model_=model,
            initial_guess=init_guess,
            x_points=x_points,
            y_points=y_points,
            sigma_points=sigma_points,
            use_errors=use_errors,
        )
        fitted_rchisqr = self.reduced_chi_squared_of_fit(fitted_model)

        if fitted_rchisqr > 10 and len(self._data) > 20 and do_halving:
            logger(
                " " * halved * 4 + f"It is unlikely that we have found the correct model... "
                f"halving the dataset to {len(self._data)//2}"
            )

            lower_data = self._data[: len(self._data) // 2]
            lower_optimizer = Optimizer(data=lower_data)
            lower_optimizer.fit_this_and_get_model_and_covariance(
                model_=model, change_shown=True, do_halving=True, halved=halved + 1
            )
            lower_rchisqr = self.reduced_chi_squared_of_fit(lower_optimizer.shown_model)
            if lower_rchisqr < fitted_rchisqr:
                fitted_model = lower_optimizer.shown_model
                fitted_cov = lower_optimizer.shown_covariance
                fitted_rchisqr = lower_rchisqr

            upper_data = self._data[len(self._data) // 2 :]
            upper_optimizer = Optimizer(data=upper_data)
            upper_optimizer.fit_this_and_get_model_and_covariance(
                model_=model, change_shown=True, do_halving=True, halved=halved + 1
            )
            upper_rchisqr = self.reduced_chi_squared_of_fit(upper_optimizer.shown_model)
            if lower_rchisqr < fitted_rchisqr:
                fitted_model = upper_optimizer.shown_model
                fitted_cov = upper_optimizer.shown_covariance
                fitted_rchisqr = upper_rchisqr

        if initial_guess is None and model_.is_submodel:
            # make the submodel realize it's a submodel
            fitted_model.set_submodel_of_zero_idx(model_.submodel_of, model_.submodel_zero_index)

        if np.any(np.isinf(fitted_cov)) and self._try_harder and halved < 1:
            try:
                logger(f"{fitted_cov=} so trying harder")
                better_cov = self.try_harder_for_cov(fitted_model)
            except OverflowError:
                pass
            else:
                fitted_cov = better_cov

        if change_shown:
            self._shown_model = fitted_model
            self._shown_covariance = fitted_cov
            self._shown_rchisqr = fitted_rchisqr

        return fitted_model, fitted_cov

    def set_estimated_std(self, model: CompositeFunction):
        logger("Optimizer.set_estimated_std(): Setting implied error bars. Resids below.")
        resids = [datum.val - model.eval_at(datum.pos) for datum in self._data]
        logger(resids)

        ss_resids = sum((res * res for res in resids))
        est_std = np.sqrt(ss_resids / (len(resids) - model.dof))
        for datum in self._data:
            datum.sigma_val = est_std

        logger(
            f"{np.finfo(np.float64).max} Optimizer.set_estimated_std() "
            f"Estimating std as {est_std} from "
            f"SSresiduals = {ss_resids} and Ndof = {len(resids) - model.dof}"
        )

    def reset_estimated_std(self, uncs: list[float]):
        for datum, unc in zip(self._data, uncs):
            datum.sigma_val = unc

    def try_harder_for_cov(self, model: CompositeFunction) -> np.ndarray:
        """When uncertainty can't be ascertained with scipy.curve_fit,
        we have to resort to other numerical methods"""

        # Should benchmark this with exact linear least squares formulae
        # in https://123.physics.ucdavis.edu/week_0_files/taylor_181-199.pdf

        uncs_y = [datum.sigma_val for datum in self._data]
        if any((unc / (abs(datum.val) + 1e-5) < 1e-5 for unc, datum in zip(uncs_y, self._data))):
            self.set_estimated_std(model)

        # to be determined
        inv_cov = np.ndarray((len(model.args), len(model.args)))

        it_limit = 30
        # find the diagonals of inv_covariance using the bisection algorithm
        while True:
            dir_bigger = []
            opt_chisqr = self.chi_squared_of_fit(model)
            opt_args = model.args.copy()
            target = opt_chisqr + 1
            for idx, arg in enumerate(opt_args):

                diff_R, better_arg_R = find_window_right(
                    self.chi_squared_of_fit, target, model, idx
                )
                if diff_R < 0:
                    model.set_arg_i(idx, better_arg_R)
                    break

                diff_L, better_arg_L = find_window_left(self.chi_squared_of_fit, target, model, idx)
                if diff_L < 0:
                    model.set_arg_i(idx, better_arg_L)
                    break

                outer_R, error_R, better_arg_R = bisect(
                    self.chi_squared_of_fit,
                    target,
                    model,
                    idx,
                    left_x=arg,
                    right_x=arg + diff_R,
                )
                if error_R > 0:
                    model.set_arg_i(idx, better_arg_R)
                    break

                outer_L, error_L, better_arg_L = bisect(
                    self.chi_squared_of_fit,
                    target,
                    model,
                    idx,
                    left_x=arg - diff_L,
                    right_x=arg,
                )
                if error_L > 0:
                    model.set_arg_i(idx, better_arg_L)
                    break

                sigma_L, sigma_R = arg - outer_L, outer_R - arg
                # sigma = 2*sigma_L*sigma_R/(sigma_L+sigma_R)
                sigma = sigma_R if sigma_R < sigma_L else sigma_L
                dir_bigger.append(1 if sigma_R < sigma_L else -1)
                # the interval left_x < x < right_x contain the point where chi^2 = chi0^2 + 1,
                # which is the definition of a 1-sigma uncertainty in the MaxLikelihood model,
                # since chi^2 ~ exp[ -(x-x0)Vinv(x-x0)/2 ]
                inv_cov[idx, idx] = 1 / sigma**2
            else:
                break

        # find the off-diagonal. This is based off chi^2 = chi0^2 + delta.Vinv.delta.
        # When chi^2 - chi0^ = 1, then delta.Vinv.delta = 1. Knowing already the diagonals
        # of Vinv, we can pick out the offdiagonals by choosing
        # delta = |D|(ei + ej). Then delta.Vinv.delta = |D|^2 (Vinv_ii + Vinv_jj + 2Vinv_ij) === 1
        #
        # Possible issues:
        #     --  it is very "wishful" that the likelihood along a
        #         particular parameter p follows the model L ~ exp(-(p-p0)^2/2sigmap^2 ).
        #         The likelihood for a step function fit is an easy counterexample
        for i, argi in enumerate(opt_args):
            for j, argj in enumerate(opt_args):

                if i >= j:
                    continue

                tmp_model = model.copy()
                inner_delta = 0
                outer_delta = max(abs(argi / 2) + 1e-5, abs(argj / 2) + 1e-5, 1e-5)

                # find domain of x that produces a range containing target
                its = 0
                while True:

                    tmp_model.set_arg_i(i, argi + dir_bigger[i] * outer_delta)
                    tmp_model.set_arg_i(j, argj + dir_bigger[j] * outer_delta)

                    right_chisqr = self.chi_squared_of_fit(tmp_model)
                    if right_chisqr > target:
                        break

                    inner_delta = outer_delta
                    outer_delta += 2 * outer_delta
                    its += 1
                    if its > it_limit:
                        logger("try_harder_for_cov: ITERATION LIMIT REACHED v2")
                        break

                diff = outer_delta - inner_delta
                its = 0
                while diff > 1e-5:

                    mid_delta = (inner_delta + outer_delta) / 2

                    tmp_model.set_arg_i(i, argi + dir_bigger[i] * mid_delta)
                    tmp_model.set_arg_i(j, argj + dir_bigger[j] * mid_delta)
                    mid_chisqr = self.chi_squared_of_fit(tmp_model)

                    if mid_chisqr < target:
                        inner_delta = mid_delta
                    else:
                        outer_delta = mid_delta
                    diff = outer_delta - inner_delta

                    its += 1
                    if its > it_limit:
                        break

                mid_delta = (inner_delta + outer_delta) / 2
                num_ij = 0.5 * dir_bigger[i] * dir_bigger[j]
                den_ij = mid_delta**2 - inv_cov[i, i] - inv_cov[j, j]
                inv_cov_ij = num_ij / den_ij
                inv_cov[i, j] = inv_cov_ij
                inv_cov[j, i] = inv_cov_ij

        cov = np.linalg.inv(inv_cov)
        logger("Optimizer.try_harder_for_cov(), invV=\n", inv_cov)
        logger("Optimizer.try_harder_for_cov(), cov=\n", cov)
        logger(f"Optimizer.try_harder_for_cov(), recalculated sigmas = {np.sqrt(np.diag(cov))}")

        if any((unc / (abs(datum.val) + 1e-5) < 1e-5 for unc, datum in zip(uncs_y, self._data))):
            self.reset_estimated_std(uncs_y)

        return cov

    def show_likelihood_across_parameter(self, model, par_idx, num_sigmas=4):

        mu = model.args[par_idx]
        cov = self.try_harder_for_cov(model)
        sigma = np.sqrt(cov[par_idx, par_idx])

        xvals = [mu + (i - num_sigmas * 10) * sigma / 10 for i in range(num_sigmas * 20)]
        yvals = [self.likelihood_with_modified_par(model, par_idx, xval) for xval in xvals]

        plt.close()
        plt.title("Likelihood across parameter")
        plt.scatter(xvals, yvals)
        plt.show()

    # changes top5
    # does not change _shown variables
    def find_best_model_for_dataset(self, status_bar=None, halved=0):

        if not halved:
            self.build_composite_function_list(status_bar=status_bar)

        x_points, y_points, sigma_points, use_errors = self.fit_setup()

        num_models = len(self._composite_function_list)
        for idx, model in enumerate(self._composite_function_list):

            fitted_model, fitted_cov = self.fit_loop(
                model_=model,
                x_points=x_points,
                y_points=y_points,
                sigma_points=sigma_points,
                use_errors=use_errors,
                info_string=f"{idx+1}/{num_models} ",
            )
            # print(f"{fitted_model.name} {fitted_cov}")
            self.query_add_to_top5(model=fitted_model, covariance=fitted_cov)
            if status_bar is not None:
                status_bar.configure(
                    text=f"   Stage 3/3: {len(self._composite_function_list):>10}"
                    f" valid models generated, {idx+1:>10} models fit."
                )
                status_bar.master.master.update()
                if status_bar["bg"] == "#010101":  # cancel code
                    break

        best_cocs = [
            self._mean_abs_coc(model, cov)
            for model, cov in zip(self._top5_models, self._top5_covariances)
        ]
        logger(
            f"\nBest models are {[m.name for m in self.top5_models]} with \n"
            f"associated reduced chi-squareds {self.top5_rchisqrs}, \n"
            f"and CoCs {best_cocs}"
        )

        logger(
            f"\nBest model is {self.top_model} "
            f"\n with args {self.top_args} += {self.top_uncs} "
            f"\n and reduced chi-sqr {self.top_rchisqr}"
        )
        self.top_model.print_sub_facts()
        self.top_model.print_tree()

        if self.top_rchisqr > 10 and len(self._data) > 20:
            # if a better model is found here it will probably
            # underestimate the actual rchisqr since
            # it will only calculate based on half the data
            logger("It is unlikely that we have found the correct model... halving the dataset")
            lower_data = self._data[: len(self._data) // 2]
            lower_optimizer = Optimizer(
                data=lower_data,
                use_functions_dict=self._use_functions_dict,
                max_functions=self._max_functions,
                regen=False,
            )
            lower_optimizer.composite_function_list = self._composite_function_list
            lower_optimizer.find_best_model_for_dataset(status_bar=status_bar)
            for l_model, l_cov in zip(
                lower_optimizer.top5_models, lower_optimizer.top5_covariances
            ):
                self.query_add_to_top5(model=l_model, covariance=l_cov)

            upper_data = self._data[len(self._data) // 2 :]
            upper_optimizer = Optimizer(
                data=upper_data,
                use_functions_dict=self._use_functions_dict,
                max_functions=self._max_functions,
                regen=False,
            )
            upper_optimizer.composite_function_list = self._composite_function_list
            upper_optimizer.find_best_model_for_dataset(status_bar=status_bar)
            for u_model, u_cov in zip(
                upper_optimizer.top5_models, upper_optimizer.top5_covariances
            ):
                self.query_add_to_top5(model=u_model, covariance=u_cov)

    # changes top5
    # does not change _shown variables
    def async_find_best_model_for_dataset(self, start=False) -> str:

        status = ""  # TODO: am I actually looking at status when this returns?
        if start:
            self._composite_generator = self.all_valid_composites_generator()

        x_points, y_points, sigma_points, use_errors = self.fit_setup(fourier_condition=start)

        batch_size = 10
        for _ in range(batch_size):

            try:
                model = next(self._composite_generator)
            except StopIteration:
                status = "Done: stop iteraton reach"
                break

            fitted_model, fitted_cov = self.fit_loop(
                model_=model,
                x_points=x_points,
                y_points=y_points,
                sigma_points=sigma_points,
                use_errors=use_errors,
                info_string=f"Async {self._gen_idx+1} ",
            )

            self.query_add_to_top5(model=fitted_model, covariance=fitted_cov)

        return status

    def find_peaks_for_gaussian(self, expected_n):
        # we assume that the peaks are "easy" to find -- that there is a zero-derivative at each one

        # there should be at least 3*expected_n datapoints
        smoothed = self.smoothed_data(n=2)  # points -= 2
        if len(self._data) > 6:
            smoothed = self.smoothed_data(n=3)  # points -= 3
        slope = self.deriv_n(data=smoothed, n=1)  # points -= 1
        if len(self._data) > 7:
            slope = self.smoothed_data(data=slope, n=1)  # points -= 1
        # so in the worst case with npoints = 6, we still have 3 points to work with

        cand_con = []
        for m0, m1 in zip(slope[:-1], slope[1:]):
            if np.sign(m0.val) != np.sign(m1.val):
                tup = (m0.pos + m1.pos) / 2, (m1.val - m0.val) / (m1.pos - m0.pos)
                cand_con.append(tup)
                logger(
                    f"New candidate at {(m0.pos + m1.pos)/2} "
                    f"with concavity {(m1.val-m0.val)/(m1.pos-m0.pos)}"
                )

        # sort the candidates by their concavity
        sorted_candidates = [cc[0] for cc in cand_con]
        if len(cand_con) > expected_n:
            sorted_cand_con = sorted(cand_con, key=lambda x: x[1])
            sorted_candidates = [cc[0] for cc in sorted_cand_con]
        elif len(cand_con) < expected_n:
            for _ in range(expected_n - len(cand_con)):
                sorted_candidates.append((self._data[0].pos + self._data[-1].pos) / 2)

        return sorted_candidates[:expected_n]

    def find_widths_for_gaussian(self, means):
        # for mean i, guess that the width is half the distance to the nearest other peak
        widths = []
        avg_bin_width = (self._data[-1].pos - self._data[0].pos) / (len(self._data) - 1)
        for mean in means:
            distances = [abs(mean - x) for x in means]
            nearest = sorted(distances)[1]  # [0] will always be a zero distance
            widths.append(nearest / 2 if nearest > avg_bin_width else avg_bin_width)
        return widths

    def find_amplitudes_for_gaussian(self, means, widths):
        amplitudes = []
        for mean in means:
            for datumlow, datumhigh in zip(self._data[:-1], self._data[1:]):
                if datumlow.pos <= mean < datumhigh.pos:
                    amplitudes.append((datumlow.val + datumhigh.val) / 2)
                    break
        avg_bin_width = (self._data[-1].pos - self._data[0].pos) / (len(self._data) - 1)
        expected_amplitude_sum = sum((datum.val for datum in self._data)) * avg_bin_width
        actual_amplitude_sum = sum(amplitudes)
        return [
            amp * expected_amplitude_sum / actual_amplitude_sum / np.sqrt(2 * np.pi * width**2)
            for (amp, width) in zip(amplitudes, widths)
        ]

    def create_cos_sin_frequency_lists(self):

        self._cos_freq_list = []
        self._sin_freq_list = []

        # need to recreate the data as a list of uniform x-interval pairs
        # (in case the data points aren't uniformly spaced)
        minx = min((datum.pos for datum in self._data))
        maxx = max((datum.pos for datum in self._data))
        if (maxx - minx) < 1e-10:
            self._cos_freq_list.append(0)
            self._sin_freq_list.append(0)
            return

        # supersample the data, with endpoints the same and one more for each interval
        x_points = np.linspace(minx, maxx, num=len(self._data) + 1)

        # very inefficient
        y_points = [self._data[0].val]
        for target in x_points[1:-1]:
            # find the two x-values in _data that surround the target value
            for idx, pos_high in enumerate([datum.pos for datum in self._data]):
                if pos_high > target:
                    # equation of the line connecting the two points is
                    # y(x) = [ y_low(x_high-x) + y_high(x-x_low) ] / (x_high - x_low)
                    x_low, y_low = self._data[idx - 1].pos, self._data[idx - 1].val
                    x_high, y_high = (self._data[idx].pos, self._data[idx].val)
                    y_points.append(
                        (y_low * (x_high - target) + y_high * (target - x_low)) / (x_high - x_low)
                    )
                    break
        y_points.append(self._data[-1].val)

        # TODO :
        #  the phase information from the FFT often gets the nature of sin/cosine wrong.
        #  Is there a way to do better?

        avg_y = sum(y_points) / len(y_points)
        zeroed_y_points = [val - avg_y for val in y_points]

        # complex fourier spectrum, with positions adjusted to frequency space
        fft_Ynu = fftshift(fft(zeroed_y_points)) / len(zeroed_y_points)
        fft_nu = fftshift(fftfreq(len(zeroed_y_points), x_points[1] - x_points[0]))

        pos_Ynu = fft_Ynu[len(fft_Ynu) // 2 :]  # the positive frequency values
        pos_nu = fft_nu[len(fft_Ynu) // 2 :]  # the positive frequencies

        # (a+bi)exp(iwt) + exp(a-bi)exp(-iwt) = 2a cos(wt) - bsin(wt)
        cos_Ynu = np.abs(pos_Ynu.real)
        sin_Ynu = np.abs(pos_Ynu.imag)

        # sort the frequencies based on the size of the cos-ness or sin-ness
        sorted_cos_nu = [nu for _, nu in sorted(zip(cos_Ynu, pos_nu), reverse=True)]
        sorted_sin_nu = [nu for _, nu in sorted(zip(sin_Ynu, pos_nu), reverse=True)]

        # limit the number of tracked frequencies to 10
        self._cos_freq_list.extend(sorted_cos_nu[:10])
        self._sin_freq_list.extend(sorted_sin_nu[:10])

        """
        I had a lot of other code to check nearby frequencies but it was flawed since it 
        didn't take into account negative amplitudes. See repos pre May 29 2024 for
        that implementation
        """

    def find_initial_guess_scaling(self, model, init_guess=None):

        # this forces a double-application of the algorithm
        if init_guess is None:
            take_one = self.find_set_initial_guess_scaling(model)
            scaling_args_no_sign = self.find_initial_guess_scaling(model, take_one)
        else:
            scaling_args_no_sign = init_guess

        # the above args, with sizes based on scaling, could each be
        # positive or negative. Find the best one (of 2^dof)
        best_rchisqr = 1e50
        best_grid_point = scaling_args_no_sign

        scaling_args_sign_list = []
        # creates list of arguments to try with all +/- sign combinations
        for idx in range(2**model.dof):
            binary_string_of_index = f"0000000000000000{idx:b}"

            new_gridpoint = scaling_args_no_sign.copy()
            for bit, arg in enumerate(new_gridpoint):
                new_gridpoint[bit] = arg * (1 - 2 * int(binary_string_of_index[-1 - bit]))
            scaling_args_sign_list.append(new_gridpoint)

        # tests each of the +/- combinations for the best fit
        # also keep running sums of weigthed points, to try their average at the end
        weighted_point = [0.0 for _ in scaling_args_no_sign]
        weighted_norm = 0
        for point in scaling_args_sign_list:
            model.set_args(*point)
            temp_rchisqr = self.reduced_chi_squared_of_fit(model)
            # if model.name == "pow_neg1+pow1":
            #     print(point, temp_rchisqr)
            #     self.show_fit(model=model, pause_on_image=True)
            weighted_point = list_sums_weights(weighted_point, point, 1, 1 / (temp_rchisqr + 1e-5))
            weighted_norm += 1e-10 if np.isnan(temp_rchisqr) else 1 / (temp_rchisqr + 1e-5)
            if temp_rchisqr < best_rchisqr:
                best_rchisqr = temp_rchisqr
                best_grid_point = point

        # this normalizes the new point
        weighted_point_norm = [xi / weighted_norm for xi in weighted_point]

        # test how good the weighted gridpoint is
        model.set_args(*weighted_point_norm)
        temp_rchisqr = self.reduced_chi_squared_of_fit(model)

        # if model.name == "pow_neg1+pow1":
        #     print(f"Weighted {weighted_point_norm} {temp_rchisqr}")

        if temp_rchisqr < best_rchisqr and model.num_trig() < 1:
            # don't want to mess up frequencies
            best_grid_point = weighted_point_norm

        # if model.name == "pow_neg1+pow1":
        #     print(f"Best {best_grid_point} {best_rchisqr}")
        model.set_args(*best_grid_point)

        return best_grid_point

    def find_set_initial_guess_scaling(self, composite: CompositeFunction):

        for child in reversed(composite.children_list):
            # reverse to bias towards more complicated
            # functions first for fourier frequency setting
            self.find_set_initial_guess_scaling(child)

        # use knowledge of scaling to guess parameter sizes from the
        # characteristic sizes in the data
        min_x = min((datum.pos for datum in self._data))
        max_x = max((datum.pos for datum in self._data))
        min_y = min((datum.val for datum in self._data))
        max_y = max((datum.val for datum in self._data))

        char_av_y = (max_y + min_y) / 2
        char_diff_y = (max_y - min_y) / 2
        char_av_x = (max_x + min_x) / 2
        char_diff_x = (max_x - min_x) / 2
        # charSpacingX =  sorted_X[1]  - sorted_X[0]
        if composite.prim.name == "pow0":
            # this typically represents a y-shift, so the average X
            # is more important than the range of x-values
            char_x = char_av_x
        elif composite.parent is not None and composite.parent.prim.name == "my_log":
            # alt-1: this arg this typically represents a log(x/x0) so the average is more important
            # but also A log(x/x0) = A log(x) - Alog(x0) ~ Alog(x) + y0
            # so A log(x0) should also scale like y, i.e.
            # char_x = char_av_x

            # y(x)/y0 = log(x/x0) implies that (taking ratio of y2/y1)
            # x0 = ( x1^(y2/y1) / x2 )^[y1/(y2-y1)]
            base = abs(abs(min_x) ** (max_y / min_y) / abs(max_x))
            exponent = abs(min_y / (max_y - min_y))
            char_x = base**exponent

            # x0 represents the spot where log(x/x0) = 0. So search for that place
            # ...
            # ...
        else:
            char_x = char_diff_x

        # defaults
        # if char_x > 0:
        #     xmul = char_x ** composite.dimension_arg
        # else:
        #     # for some reason negative char_x implies char_diff_x is more important
        #     xmul = char_diff_x ** composite.dimension_arg
        xmul = char_x**composite.dimension_arg
        ymul = 1

        # overrides
        if composite.parent is None:
            if composite.prim.name[:2] != "n_" and composite.prim.name != "sum_":
                ymul = char_diff_y
            else:
                pass
        elif composite.parent.prim.name == "sum_" and composite.parent.parent is None:
            ymul = char_diff_y
        elif "cos" in composite.parent.prim.name:
            # in cos( Aexp(Lx) ), for small x the inner composition goes like A + ALx
            # = f(0) + x f'(0) and here f'(0) should correspond to the largest fourier component
            # problem is that this answer should be independent of the initial parameter A...
            # it only works from scratch since we set A = 1 for new composites
            dy = composite.eval_at(2e-5) - composite.eval_at(1e-5)
            dx = 1e-5 * composite.prim.arg
            slope_at_zero = dy / dx
            if 1e-5 < abs(slope_at_zero) < 1e5:
                if len(self._cos_freq_list_dup) > 0:
                    logger(f"Using cosine ang. frequency {2*np.pi*self._cos_freq_list_dup[0]}")
                    xmul = (2 * np.pi * self._cos_freq_list_dup.pop(0)) / slope_at_zero
                    logger(f"   so with {slope_at_zero=}, {xmul=}")

                else:  # misassigned cosine frequency
                    xmul = (2 * np.pi * self._sin_freq_list_dup.pop(0)) / slope_at_zero

        elif "sin" in composite.parent.prim.name:
            dy = composite.eval_at(2e-5) - composite.eval_at(1e-5)
            dx = 1e-5 * composite.prim.arg
            slope_at_zero = dy / dx
            if 1e-5 < abs(slope_at_zero) < 1e5:
                if len(self._sin_freq_list_dup) > 0:
                    logger(f"Using sine ang. frequency {2*np.pi*self._sin_freq_list_dup[0]}")
                    xmul = (2 * np.pi * self._sin_freq_list_dup.pop(0)) / slope_at_zero
                    logger(f"   so with {slope_at_zero=}, {xmul=}")
                else:  # misassigned cosine frequency
                    xmul = (2 * np.pi * self._cos_freq_list_dup.pop(0)) / slope_at_zero

        # if composite.parent is not None:
        #     if composite.parent.prim.name == "sum_" and composite.prim.name == "pow1":
        #         print(f"{composite.parent} {xmul=} {ymul=}")
        composite.prim.arg = xmul * ymul

        return composite.get_args()

    def make_fit_image(self, model=None):

        x_points = []
        y_points = []
        sigma_x_points = []
        sigma_y_points = []
        upper_bar = []
        lower_bar = []

        for datum in self._data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            sigma_x_points.append(datum.sigma_pos)
            sigma_y_points.append(datum.sigma_val)
            upper_bar = [y + dy for y, dy in zip(y_points, sigma_y_points)]
            lower_bar = [y - dy for y, dy in zip(y_points, sigma_y_points)]

        smooth_x_for_fit = np.linspace(min(x_points), max(x_points), 4 * len(x_points))

        if model is not None:
            plot_model = model.copy()
        else:
            plot_model = self._shown_model

        fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]

        # plt.close()
        plt.figure(
            facecolor=(112 / 255, 146 / 255, 190 / 255),
            figsize=(6.4, 4.8),
            dpi=100 + int(np.log10(len(x_points))),
        )
        plt.errorbar(
            x_points,
            y_points,
            xerr=sigma_x_points,
            yerr=sigma_y_points,
            fmt="o",
            color="k",
        )
        plt.plot(smooth_x_for_fit, fit_vals, "-", color="r")

        plt.xlabel("x")
        plt.ylabel("y")
        axes: plt.axes = plt.gca()

        pf.zero_out_axes(axes)
        pf.set_xaxis_format_linear(axes, x_points)
        pf.set_yaxis_format_linear(axes, y_points)

        axes.set_facecolor((112 / 255, 146 / 255, 190 / 255))

        pf.fix_axes_labels(axes, min(x_points), max(x_points), min(lower_bar), max(upper_bar), "x")

        my_string_iobytes = BytesIO()
        plt.savefig(my_string_iobytes, format="png")
        my_string_iobytes.seek(0)
        my_base64_png_data = b64encode(my_string_iobytes.read()).decode()
        return my_base64_png_data

    def show_fit(self, model=None, pause_on_image=False):

        x_points = []
        y_points = []
        sigma_x_points = []
        sigma_y_points = []

        for datum in self._data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            sigma_x_points.append(datum.sigma_pos)
            sigma_y_points.append(datum.sigma_val)

        if model is not None:
            plot_model = model.copy()
        else:
            plot_model = self._shown_model

        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor((112 / 255, 146 / 255, 190 / 255))
        plt.errorbar(
            x_points,
            y_points,
            xerr=sigma_x_points,
            yerr=sigma_y_points,
            fmt="o",
            color="k",
        )
        if plot_model is not None:
            smooth_x_for_fit = np.linspace(x_points[0], x_points[-1], 4 * len(x_points))
            fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]
            plt.plot(smooth_x_for_fit, fit_vals, "-", color="r")
        plt.xlabel("x")
        plt.ylabel("y")
        axes = plt.gca()
        if axes.get_xlim()[0] > 0:
            axes.set_xlim([0, axes.get_xlim()[1]])
        if axes.get_ylim()[0] > 0:
            axes.set_ylim([0, axes.get_ylim()[1]])
        axes.set_facecolor((112 / 255, 146 / 255, 190 / 255))

        plt.show(block=pause_on_image)

    def inferred_error_bar_size(self) -> float:
        return 1
        # return (
        # + max([datum.val for datum in self._data])
        # - min([datum.val for datum in self._data])
        # )/10
        # return np.array([datum.val - model.eval_at(datum.pos) for datum in self._data]).std()

    def chi_squared_of_fit(self, model: CompositeFunction) -> float:

        if any((datum.sigma_val < 1e-5 for datum in self._data)):
            sumsqr = sum(((model.eval_at(datum.pos) - datum.val) ** 2 for datum in self._data))
            return sumsqr / self.inferred_error_bar_size() ** 2

        return sum(
            (
                (model.eval_at(datum.pos) - datum.val) ** 2 / datum.sigma_val**2
                for datum in self._data
            )
        )

    def likelihood(self, model: CompositeFunction) -> float:
        return np.exp(-self.chi_squared_of_fit(model) / 2)

    def likelihood_with_modified_par(
        self, model: CompositeFunction, par_idx: int, new_par: float
    ) -> float:
        tmp_model = model.copy()
        tmp_pars = tmp_model.args
        tmp_pars[par_idx] = new_par
        tmp_model.args = tmp_pars
        lhood = self.likelihood(tmp_model)
        return lhood

    def reduced_chi_squared_of_fit(self, model) -> float:
        k = model.dof
        N = len(self._data)
        return self.chi_squared_of_fit(model) / (N - k) if N > k else 1e5

    def _multiplier_for_high_frequencies(self, model) -> float:
        """
        Compares the chosen frequency to the inferred grid spacing of the data.
        If the frequency is too high to reasonably fit the grid, the model
        is punished with a multiplier, returned here.
        """
        avg_grid_spacing = (self._data[-1].pos - self._data[0].pos) / len(self._data)

        for idx in range(model.num_nodes()):
            node = model.get_node_with_index(idx)
            if node.parent is not None and node.parent.prim.name in [
                "my_sin",
                "my_cos",
            ]:
                if node.prim.arg > 10 * avg_grid_spacing:
                    logger("LIAR! FREQUENCY EXTRAVAGENT")
                    return (node.prim.arg / avg_grid_spacing) ** 2
        return 1

    def _criterion_modified(self, model, cov):
        """
        Returns the chosen criterion plus additional penalties
        for high covariance and large frequencies
        """
        frequency_multiplier = self._multiplier_for_high_frequencies(model)
        covariance_adjustment = self._mean_abs_coc(model, cov)
        base_criterion = self._criterion(model)
        return base_criterion * frequency_multiplier + covariance_adjustment

    @staticmethod
    def _mean_abs_coc(model, cov):
        """Returns the sum of the absolute values in the Coefficient of Covariance matrix"""
        N = len(model.args)
        cc = coefficient_of_covariance(cov, model.args)
        return np.sum(np.abs(cc)) / N**2

    def r_squared(self, model):
        mean = np.mean([datum.val for datum in self._data])
        variations_from_mean = [(datum.val - mean) ** 2 for datum in self._data]
        variations_from_fit = [(datum.val - model.eval_at(datum.pos)) ** 2 for datum in self._data]
        variance_data = np.mean(variations_from_mean)
        variance_fit = np.mean(variations_from_fit)

        return 1 - variance_fit / variance_data

    def akaike_criterion(self, model):
        # the AIC is equivalent, for normally distributed residuals, to the least chi squared
        k = model.dof
        # N = len(self._data)
        AIC = self.chi_squared_of_fit(model) + 2 * k  # we take chi^2 = -2logL + C, discard the C
        return AIC

    def akaike_criterion_corrected(self, model):
        # correction for small datasets, fixes overfitting
        k = model.dof
        N = len(self._data)
        if N > k + 1:
            return self.akaike_criterion(model) + 2 * k * (k + 1) / (N - k - 1)
        return 1e5

    def bayes_criterion(self, model):
        # the same as AIC but penalizes additional parameters more heavily for larger datasets
        k = model.dof
        N = len(self._data)
        BIC = self.chi_squared_of_fit(model) + k * np.log(N)
        return BIC

    def hannan_quinn_criterion(self, model):
        # agrees with AIC at small datasets, but punishes less strongly than Bayes at large N
        k = model.dof
        N = len(self._data)
        if N > 1:
            return self.chi_squared_of_fit(model) + 2 * k * np.log(np.log(N))
        return 1e5

    def smoothed_data(self, data=None, n=1) -> list[Datum1D]:

        logger("Using smoothed data")

        return_data = []
        if data is None:
            if n <= 1:
                data_to_smooth = [copy(datum) for datum in self.average_data()]
            else:
                data_to_smooth = self.smoothed_data(n=n - 1)
        else:
            data_to_smooth = data

        for idx, datum in enumerate(data_to_smooth[:-1]):
            new_pos = (data_to_smooth[idx + 1].pos + data_to_smooth[idx].pos) / 2
            new_val = (data_to_smooth[idx + 1].val + data_to_smooth[idx].val) / 2
            # propagation of uncertainty
            new_sigma_pos = np.sqrt(
                (data_to_smooth[idx + 1].sigma_pos / 2) ** 2
                + (data_to_smooth[idx].sigma_pos / 2) ** 2
            )
            new_sigma_val = np.sqrt(
                (data_to_smooth[idx + 1].sigma_val / 2) ** 2
                + (data_to_smooth[idx].sigma_val / 2) ** 2
            )
            return_data.append(
                Datum1D(
                    pos=new_pos,
                    val=new_val,
                    sigma_pos=new_sigma_pos,
                    sigma_val=new_sigma_val,
                )
            )

        return return_data

    def deriv_n(self, data, n=1) -> list[Datum1D]:

        # this assumes that data is sequential, i.e. that there are
        # no repeated measurements for each x position
        return_deriv = []
        if n <= 1:
            data_to_diff = data
        else:
            data_to_diff = self.deriv_n(data=data, n=n - 1)
        for idx, _ in enumerate(data_to_diff[:-2]):
            new_deriv = (data_to_diff[idx + 2].val - data_to_diff[idx].val) / (
                data_to_diff[idx + 2].pos - data_to_diff[idx].pos
            )
            new_pos = (data_to_diff[idx + 2].pos + data_to_diff[idx].pos) / 2

            # propagation of uncertainty
            new_sigma_pos = np.sqrt(
                (data_to_diff[idx + 2].sigma_pos / 2) ** 2 + (data_to_diff[idx].sigma_pos / 2) ** 2
            )
            new_sigma_deriv = (1 / (data_to_diff[idx + 2].pos - data_to_diff[idx].pos)) * np.sqrt(
                data_to_diff[idx + 2].sigma_val ** 2 + data_to_diff[idx].sigma_val ** 2
            ) + new_deriv**2 * (
                data_to_diff[idx + 2].sigma_pos ** 2 + data_to_diff[idx].sigma_pos ** 2
            )
            return_deriv.append(
                Datum1D(
                    pos=new_pos,
                    val=new_deriv,
                    sigma_pos=new_sigma_pos,
                    sigma_val=new_sigma_deriv,
                )
            )

        return return_deriv

    # this is only used to find an initial guess for the data, and so
    # more sophisticated techniques like weighted means is not reqired
    def average_data(self) -> list[Datum1D]:

        # get means and number of pos-instances into dict
        sum_val_dict: dict[float, float] = {}
        num_dict: dict[float, int] = {}
        for datum in self._data:
            sum_val_dict[datum.pos] = sum_val_dict.get(datum.pos, 0) + datum.val
            num_dict[datum.pos] = num_dict.get(datum.pos, 0) + 1

        mean_dict = {}
        for ikey, isum in sum_val_dict.items():
            mean_dict[ikey] = isum / num_dict[ikey]

        propagation_variance_x_dict: dict[float, float] = {}
        propagation_variance_y_dict: dict[float, float] = {}
        sample_variance_y_dict: dict[float, float] = {}
        for datum in self._data:
            # average the variances
            propagation_variance_x_dict[datum.pos] = (
                propagation_variance_x_dict.get(datum.pos, 0) + datum.sigma_pos**2
            )
            propagation_variance_y_dict[datum.pos] = (
                propagation_variance_y_dict.get(datum.pos, 0) + datum.sigma_val**2
            )
            sample_variance_y_dict[datum.pos] = (
                sample_variance_y_dict.get(datum.pos, 0) + (datum.val - mean_dict[datum.pos]) ** 2
            )

        averaged_data = []
        for key, val in mean_dict.items():
            sample_uncertainty_squared = (
                sample_variance_y_dict[key] / (num_dict[key] - 1) if num_dict[key] > 1 else 0
            )
            propagation_uncertainty_squared = propagation_variance_y_dict[key] / num_dict[key]
            ratio = (
                sample_uncertainty_squared
                / (sample_uncertainty_squared + propagation_uncertainty_squared)
                if propagation_uncertainty_squared > 0
                else 1
            )

            # interpolates smoothly between 0 uncertainty in data points
            # (so all uncertainty comes from sample spread)
            # to the usual uncertainty coming from both the data and the spread
            # TODO: see
            #  https://stats.stackexchange.com/questions/454120/
            #  how-can-i-calculate-uncertainty-of-the-mean-of-a-set-of-samples-with-different-u
            # for a more rigorous treatment
            effective_uncertainty_squared = (
                ratio * sample_uncertainty_squared + (1 - ratio) * propagation_uncertainty_squared
            )

            averaged_data.append(
                Datum1D(
                    pos=key,
                    val=val,
                    sigma_pos=np.sqrt(propagation_variance_x_dict[key]),
                    sigma_val=np.sqrt(effective_uncertainty_squared),
                )
            )
        return sorted(averaged_data)

    def load_default_functions(self):
        self._primitive_function_list.extend(
            [PrimitiveFunction.built_in("pow0"), PrimitiveFunction.built_in("pow1")]
        )

    def load_non_defaults_to_primitive_function_list(self):
        for key, use_function in self._use_functions_dict.items():
            if key == "cos(x)" and use_function:
                self._primitive_function_list.append(PrimitiveFunction.built_in("cos"))
            if key == "sin(x)" and use_function:
                self._primitive_function_list.append(PrimitiveFunction.built_in("sin"))
            if key == "exp(x)" and use_function:
                self._primitive_function_list.append(PrimitiveFunction.built_in("exp"))
            if key == "log(x)" and use_function:
                self._primitive_function_list.append(PrimitiveFunction.built_in("log"))
            if key == "1/x" and use_function:
                self._primitive_function_list.append(PrimitiveFunction.built_in("pow_neg1"))
            # custom functions are also loaded, but elsewhere, using add_primitive_to_list()

    @staticmethod
    def all_defaults_on_dict():
        all_functions_dict = {
            "cos(x)": True,
            "sin(x)": True,
            "exp(x)": True,
            "log(x)": True,
            "1/x": True,
        }
        return all_functions_dict


def std(cov):
    """
    Returns the standard deviations of a covariance matrix
    """
    return list(np.sqrt(np.diagonal(cov)))


def correlation_from_covariance(cov_mat: np.ndarray) -> np.ndarray:
    """
    Returns the correlation matrix, with the covariance matrix as input
    """
    v = np.sqrt(np.diag(cov_mat))
    outer_v = np.outer(v, v)
    correlation = cov_mat / outer_v
    correlation[cov_mat == 0] = 0
    return correlation


def coefficient_of_covariance(cov_mat: np.ndarray, mean_list: list[float]) -> np.ndarray:
    """
    Coefficient of covariance is defined as
    cc_ij = cov_ij / (mu_i mu_j)
    I.e. it is the size of the covariance relative to the related means
    """
    cc = cov_mat.copy()
    for idx, (cov_i, mean_i) in enumerate(zip(cov_mat, mean_list)):
        for jdx, (cov_ij, mean_j) in enumerate(zip(cov_i, mean_list)):
            cc[idx][jdx] = cov_ij / (mean_i * mean_j)
    return cc


def sum_sqr_off_diag(cc: np.ndarray) -> float:
    """
    Calculates and returns the sum of squared off-diagonal elements in a matrix.
    The variable is called `cc` because the intended usage is to have the coefficient-of-covariance
    matrix as input
    """
    sumsqr = 0
    for (idx, jdx), cc_ij in np.ndenumerate(cc):
        if idx < jdx:
            sumsqr += cc_ij**2
    return sumsqr


def list_sums_weights(l1, l2, w1, w2) -> list[float]:
    """
    Returns the weighted sum of two lists. e.g. if l1 and l2 are vectors and w1, w2 are scalars,
    then this returns L = w1 l1 + w2 L2
    """
    w1 = 0 if np.isnan(w1) else w1
    w2 = 0 if np.isnan(w2) else w2
    if len(l1) != len(l2):
        return []
    sum_list = []
    for val1, val2 in zip(l1, l2):
        sum_list.append(val1 * w1 + val2 * w2)
    return sum_list
