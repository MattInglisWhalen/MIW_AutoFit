
# default libraries
import math
import re as regex
from collections import defaultdict
import codeop

# external libraries
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt
import numpy as np

# internal classes
from autofit.src.datum1D import Datum1D
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.composite_function import CompositeFunction


"""
This Optimizer class creates a list of possible fit functions. 
Using scipy's curve-fit function, each of the possible fit functions are fit to the data provided.
This class will then determine which of the functions is the best model for the data, using the reduced chi^2 statistic
as a discriminator. The model which achieves a reduced chi-squared closest to 1 [in practice, the model which minimizes
( log rchi^2 )^2 ] is chosen to be the "best" model
"""
DEBUG = False

class Optimizer:

    def __init__(self, data=None, use_functions_dict = None, max_functions=3):

        print("New optimizer created")

        # datasets, which are lists of Datum1D instances
        self._data = []             # the raw datapoints of (x,y), possibly with uncertainties
                                    # May also represent a binned point if the input file is a list of x_values only
        self._averaged_data = []    # the model optimizer requires approximate slope knowledge,
                                    # so multiple measurements at each x-value are collapsed into a single value

        # fit results
        self._shown_model = None          # a CompositeFunction
        self._shown_covariance = None
        self._shown_rchisqr = 1e5

        # top 5 models and their data
        self._top5_models = []
        self._top5_covariances = []
        self._top5_rchisqrs = [1e5]  # [1e5  for _ in range(5)]

        # function construction parameters
        self._max_functions = max_functions
        self._primitive_function_list = []
        self._primitive_names_list = []

        # useful auxiliary varibles
        self._temp_function : CompositeFunction = None      # for ?
        self._cos_freq_list = None
        self._sin_freq_list = None
        self._cos_freq_list_dup = None
        self._sin_freq_list_dup = None

        if data is not None :
            self._data = sorted(data)  # list of Datum1D. Sort makes the data monotone increasing in x-value
            self.average_data()

        self.load_default_functions()
        if use_functions_dict is None:
            use_functions_dict = {}
        self._use_functions_dict = use_functions_dict
        self.load_non_defaults_from(use_functions_dict)

        self._composite_function_list = []
        self._composite_generator = None
        self._gen_idx = -1

        self._regen_composite_flag = True

    def __repr__(self):
        pass

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
    def shown_rchisqr(self):
        return self._shown_rchisqr
    @shown_rchisqr.setter
    def shown_rchisqr(self, val):
        self._shown_rchisqr = val

    @property
    def top5_models(self):
        return self._top5_models
    @top5_models.setter
    def top5_models(self, other):
        self._top5_models = other
    @property
    def top_model(self):
        return self.top5_models[0]
    @property
    def top5_args(self):
        return [model.args for model in self.top5_models]
    @property
    def top_args(self):
        return self.top5_args[0]
    @property
    def top5_uncertainties(self):
        return [list(np.sqrt(np.diagonal(covariance))) for covariance in self._top5_covariances]
    @property
    def top_uncs(self):
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
    def top5_names(self):
        return [model.name for model in self._top5_models]
    @property
    def top_name(self):
        return self.top5_names[0]
    @property
    def top5_rchisqrs(self):
        return self._top5_rchisqrs
    @top5_rchisqrs.setter
    def top5_rchisqrs(self, other):
        self._top5_rchisqrs = other
    @property
    def top_rchisqr(self):
        return self.top5_rchisqrs[0]

    @property
    def composite_function_list(self):
        return self._composite_function_list
    @composite_function_list.setter
    def composite_function_list(self, comp_list):
        self._composite_function_list = comp_list

    def update_opts(self, use_functions_dict, max_functions: int):
        self._use_functions_dict = use_functions_dict
        self._max_functions = max_functions
        self._regen_composite_flag = True

    # def update_top5_rchisqrs_for_new_data_single_model(self, new_data, chosen_model):
    #     self.set_data_to(new_data)
    #     for idx, model in enumerate(self.top5_models) :
    #         if model.name == chosen_model.name :
    #             self.top5_rchisqrs[idx] = self.reduced_chi_squared_of_fit(model)
    def update_top5_rchisqrs_for_new_data(self, new_data):
        self.set_data_to(new_data)
        for idx, model in enumerate(self.top5_models) :
            self.top5_rchisqrs[idx] = self.reduced_chi_squared_of_fit(model)

    # changes top5 lists, does not change _shown variables
    def query_add_to_top5(self, model: CompositeFunction, covariance):

        rchisqr = self.reduced_chi_squared_of_fit(model)
        print(f"Querying {rchisqr} for {model.name}")
        if rchisqr > self.top5_rchisqrs[-1] :
            return

        for idx, (topper, chisqr) in enumerate(zip( self.top5_models[:],self.top5_rchisqrs[:] )):
            if model.longname == topper.longname :
                if chisqr <= rchisqr :
                    return
                del self.top5_models[idx]
                del self.top5_covariances[idx]
                del self.top5_rchisqrs[idx]
                # self.top5_models.remove( self.top5_models[idx] )
                # self.top5_covariances.remove( self.top5_covariances[idx] )
                # self.top5_rchisqrs.remove( self.top5_rchisqrs[idx])
                break

        # print(f"Passed basic check, trying to add {rchisqr} to {self._top5_rchisqrs}")
        for idx, chi_sqr in enumerate(self._top5_rchisqrs[:]) :
            if self.modified_chi_sqr_for_believability(model) < self._top5_rchisqrs[idx] :
                self.top5_models.insert(idx, model.copy())
                self.top5_covariances.insert(idx, covariance)
                self.top5_rchisqrs.insert(idx, rchisqr)

                self.top5_models = self._top5_models[:5]
                self.top5_covariances = self._top5_covariances[:5]
                self.top5_rchisqrs = self._top5_rchisqrs[:5]

                print(f"New top {idx} with red_chisqr={rchisqr:.2F}: "
                      f"{model=} with pars={model.args} \u00B1 {std(covariance)}")
                model.print_tree()

                # also check any parameters for being "equivalent" to zero. If so, remove the d.o.f.
                # and add the new function to the top5 list
                for ndx, (arg, unc) in enumerate( zip(model.args,std(covariance)) ) :
                    if abs(arg) < 2*unc :  # 95% confidence level arg is not different from zero
                        print("Zero arg detected: new trimmed model is")
                        reduced_model = model.submodel_without_node_idx(ndx)
                        if ( reduced_model is None or self.fails_rules(reduced_model) or
                                (reduced_model.prim.name == "sum_" and reduced_model.num_children() == 0) ):
                            print("Can't reduce the model")
                            continue
                        reduced_model.set_submodel_of_zero_idx(model,ndx)
                        reduced_model.print_tree()
                        # this method changes shown models
                        improved_reduced, improved_cov = self.fit_this_and_get_model_and_covariance(
                                                            model_=reduced_model,initial_guess=reduced_model.args
                                                         )
                        self.query_add_to_top5(model=improved_reduced, covariance=improved_cov)
                        break
                return

    # update for which reason?
    # def update_resort_top5_rchisqr(self):
    #     self._top5_rchisqrs = [1e5 for _ in self._top5_rchisqrs]
    #     for model, cov in zip(self._top5_models[:], self._top5_covariances[:]) :
    #         self.query_add_to_top5(model=model,covariance=cov)


    def composite_function_generator(self, depth):

        for model in CompositeFunction.built_in_list() :
            yield model

        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        if depth <= 1 :
            for iprim in self._primitive_function_list:
                new_comp = CompositeFunction(prim_=iprim)
                yield new_comp

        else :
            head_gen = self.composite_function_generator( depth = depth-1 )
            for icomp in head_gen :
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list:
                        new_comp = icomp.copy()
                        new_comp.get_node_with_index(idescendent).add_child(iprim)
                        new_comp.build_longname()
                        yield new_comp

                        new_mul = icomp.copy()
                        new_mul.get_node_with_index(idescendent).add_younger_brother(iprim)
                        new_mul.build_longname()
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
    def build_composite_function_list(self):
        # the benefit of using this is that you can generate it once, and if the options don't change you don't need
        # to generate it again. Follow that logic
        if not self._regen_composite_flag :
            return
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
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list[:]:

                        new_comp = icomp.copy()
                        sum_node = new_comp.get_node_with_index(idescendent)
                        # the naming already sorts the multiplication and summing parts by descrinding order,
                        # so why not just build that in
                        if sum_node.num_children() > 0 and iprim.name > sum_node.children_list[-1].prim.name  :
                            pass
                        else:
                            if sum_node.prim.name in ["my_sin","my_cos"] and iprim.name in ["my_sin","my_cos"] :
                                pass  # speedup for double trig
                            elif sum_node.prim.name == "my_log" and iprim.name in ["my_log","my_exp"] :
                                pass
                            elif sum_node.prim.name == "my_exp" and iprim.name == "my_exp" :
                                pass
                            else:
                                sum_node.add_child(iprim, update_name=True)
                                new_list.append(new_comp)

                        new_mul = icomp.copy()
                        mul_node = new_mul.get_node_with_index(idescendent)
                        if mul_node.prim.name >= iprim.name :
                            if mul_node.prim.name == "my_exp" and iprim.name == "my_exp" :
                                pass # speedup for multiplied exps
                            else :
                                mul_node.add_younger_brother(iprim, update_name=True)
                                new_list.append(new_mul)

            self._composite_function_list.extend( new_list )
            last_list = new_list

        for comp in CompositeFunction.built_in_list():
            self._composite_function_list.append(comp.copy())

        # prepend the current top 5 models

        self.trim_composite_function_list()
        print(f"After trimming list: (len={len(self._composite_function_list)})")
        for icomp in self._composite_function_list:
            print(icomp)
        print("|----------------\n")
    def trim_composite_function_list(self):
        # Finds and removes duplicates
        # Performs basic algebra to recognize simpler forms with fewer parameters
        # E.g. Functions like pow0(pow0) and pow1(pow1 + pow1) have fewer arguments than the naive calculation expects

        # Only run this at the end of the tree generation: if you run this in intermediate steps,
        # you will miss out on some functions possibilities

        num_comps = len(self._composite_function_list[:])
        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        # use regex to trim based on rules applied to composite names
        for idx, icomp in enumerate(self._composite_function_list[:]) :

            if idx % 50 == 0 :
                print(f"{idx}/{num_comps}")

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
            if child.prim.name == "pow1" and child.num_children() > 0 and child.younger_brother is None and child.older_brother is None :
                return 97



        # composition of a constant function is wrong
        if regex.search(f"pow0\(", name):
            return 2
        if regex.search(f"\(pow0\)", name):
            return 3
        # composition of powers with no sum is wrong -- not strictly true -- think of log^2(Ax+B)
        if regex.search(f"pow[0-9]\(pow[a-z0-9_·]*\)", name):
            return 5


        # repeated reciprocal is wrong
        if regex.search(f"pow_neg1\(pow_neg1\)", name):
            return 7
            # and deeper
        if regex.search(f"pow_neg1\(pow_neg1\([a-z0-9_+]*]\)\)", name):
            return 7

        # trivial reciprocal is wrong
        if regex.search(f"pow_neg1\(pow1\)", name):
            return 11



        # sum of pow1 with composition following is wrong
        if regex.search(f"\+pow1\(", name):
            return 19

        # sum of the same function (without further composition) is wrong
        for prim_name in self._primitive_names_list:
            if regex.search(f"{prim_name}[a-z0-9_+]*{prim_name}", name):
                return 23



        # pow0+log(...) is a duplicate since A + Blog( Cf(x) ) = B log( exp(A/B) Cf(x) ) = log(f)
        if regex.search(f"pow0\+my_log\([a-z0-9_]*\)", name) \
                or regex.search(f"my_log\([a-z0-9_]*\)\+pow0", name):
            return 37



        # trivial composition should just be straight sums
        if regex.search(f"\(pow1\(", name) or regex.search(f"\+pow1\(", name):
            return 47

        # more more exp algebra: Alog( Bexp(Cx) ) = AlogB + ACx = pow1(pow0+pow1)
        if regex.search(f"my_log\(my_exp\(pow1\)\)", name):
            return 53

        # reciprocal exponent is already an exponent pow_neg1(exp) == my_exp(pow1)
        if regex.search(f"pow_neg1\(my_exp\)", name):
            return 61
        # and again but deeper
        if regex.search(f"pow_neg1\(my_exp\([a-z0-9_+]*\)\)", name):
            return 61

        # log of "higher" powers is the same as log(pow1)
        if regex.search(f"my_log\(pow[a-z2-9_]*\)", name) or regex.search(f"my_log\(pow_neg1\)", name):
            return 67

        # exp(pow0+...) is just exp(...)
        if regex.search(f"my_exp\(pow0", name) or regex.search(f"my_exp\([a-z0-9_+·]*pow0", name):
            return 71

        # log(1/f+g) or log( (f+g)^n )is the same as log(f+g)
        if regex.search(f"my_log\(pow[a-z0-9_]*\([a-z0-9_+]*\)\)", name):
            return 73

        # pow3(exp(...)) is just exp(3...)
        if regex.search(f"pow[a-z0-9_]*\(my_exp\([a-z0-9_+]*\)\)", name):
            return 83

        """
        Multiplicative rules
        """
        if regex.search(f"pow0·", name) or regex.search(f"·pow0", name):
            return 1000 + 2

        if regex.search(f"my_exp·my_exp",name):
            return 1000 + 3

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
                     "my_exp(my_log)+my_exp(my_log)"]

        for good in good_list :
            if name == good and remove_flag:
                print(f"\n\n>>> Why did we remove {name=} at {remove_flag=} <<<\n\n")
                raise SystemExit

        if name == "pow1(pow0+pow0)" and not remove_flag:
            print(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=23 <<<\n\n")
            raise SystemExit

        if name == "pow1·pow1(pow1·pow1)" and not remove_flag:
            print(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=?? <<<\n\n")
            raise SystemExit

        if name == "my_exp(pow1)·my_exp·pow1" and not remove_flag:
            print(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=43 <<<\n\n")
            print(icomp.has_double_expness())
            raise SystemExit

        if (name == "pow1·my_exp(my_exp)" or name == "pow1·my_exp(pow1·my_exp)") and not remove_flag:
            print(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=29 <<<\n\n")
            print(icomp.has_double_expness())
            raise SystemExit

        if name == "my_exp(pow1)·my_exp(pow1)" :
            print(f"\n\n>>> Why did we NOT remove {name=} at remove_flag=?? <<<\n\n")
            raise RuntimeWarning

        return remove_flag

    def add_primitive_to_list(self, name, functional_form) -> str:
        # For adding one's own Primitive Functions to the built-in list

        if self._use_functions_dict["custom"] != 1 :
            return ""
        if name in [prim.name for prim in self._primitive_function_list ] :
            return ""

        # print(f"\n{name} {functional_form}")
        if regex.search("\\\\",functional_form) or regex.search("\n",functional_form)\
                or regex.search("\s",functional_form) :
            return "Stop trying to inject code"


        code_str  = f"def {name}(x,arg):\n"
        code_str += f"    return arg*{functional_form}\n"
        code_str += f"new_prim = PrimitiveFunction(func={name})\n"
        code_str += f"dict = PrimitiveFunction.built_in_dict()\n"
        code_str += f"dict[\"{name}\"] = new_prim\n"

        # print(f">{code_str}<")
        exec(code_str)

        # for key, item in PrimitiveFunction.built_in_dict().items() :
        #     print(f"{key}, {item}, {item.eval_at(1.5)}")

        self._primitive_function_list.append( PrimitiveFunction.built_in(name) )

        return ""
    def set_data_to(self, other_data):
        self._data = sorted(other_data)
        self.average_data()

    # does not change self._data
    def fit_setup(self, fourier_condition = True) -> (list[float], list[float], list[float], bool):

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

        print(f"\nFitting {model=}")
        print(model.tree_as_string_with_dimensions())

        if model.num_trig() > 0:
            self._cos_freq_list_dup = self._cos_freq_list.copy()
            self._sin_freq_list_dup = self._sin_freq_list.copy()

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

        print(f"{info_string}Scaling guess: {initial_guess}")

        # Next, find a better guess by relaxing the error bars on the data
        # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
        # and so relaxed error bars help point towards global minima
        try:
            better_guess, better_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                        p0=initial_guess, maxfev=5000, method='lm')
        except RuntimeError:
            print("Couldn't find optimal parameters for better guess.")
            model.args = list(initial_guess)
            return model, np.array([1e5 for _ in range(len(initial_guess)**2)]).reshape(len(initial_guess),len(initial_guess))
        print(f"{info_string}Better guess: {better_guess}")

        # Finally, use the better guess to find the true minimum with the true error bars
        try:
            np_pars, np_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                        sigma=sigma_points, absolute_sigma=use_errors,
                                        p0=better_guess, maxfev=5000)
        except RuntimeError:
            print("Couldn't find optimal parameters for final fit.")
            model.args = list(better_guess)
            return model, better_cov

        print(f"{info_string}Final guess: {np_pars}")
        model.args = list(np_pars)
        return model, np_cov

    # does not change input model_
    # changes _shown variables
    # does not change top5 lists
    def fit_this_and_get_model_and_covariance(self, model_ : CompositeFunction,
                                                    initial_guess = None,
                                                    do_halving = False, halved = 0):

        model = model_.copy()

        x_points, y_points, sigma_points, use_errors = self.fit_setup()

        fitted_model, fitted_cov = self.fit_loop(model_=model, initial_guess=initial_guess,
                                                 x_points=x_points,y_points=y_points,sigma_points=sigma_points,
                                                 use_errors=use_errors)

        self._shown_model = fitted_model
        self._shown_covariance = fitted_cov
        self._shown_rchisqr = self.reduced_chi_squared_of_fit(fitted_model)

        if self._shown_rchisqr > 5 and len(self._data) > 20 and do_halving:
            print(" "*halved*4 + f"It is unlikely that we have found the correct model... "
                                 f"halving the dataset to {math.floor(len(self._data)/2)}")

            lower_data = self._data[:math.floor(len(self._data)/2)]
            lower_optimizer = Optimizer(data=lower_data)
            lower_optimizer.fit_this_and_get_model_and_covariance(model_=model, do_halving=True, halved=halved + 1)
            lower_rchisqr = self.reduced_chi_squared_of_fit(lower_optimizer.shown_model)
            if lower_rchisqr < self._shown_rchisqr :
                self._shown_model = lower_optimizer.shown_model
                self._shown_covariance = lower_optimizer.shown_covariance
                self._shown_rchisqr = lower_rchisqr

            upper_data = self._data[math.floor(len(self._data)/2):]
            upper_optimizer = Optimizer(data=upper_data)
            upper_optimizer.fit_this_and_get_model_and_covariance(model_=model, do_halving=True, halved=halved + 1)
            upper_rchisqr = self.reduced_chi_squared_of_fit(upper_optimizer.shown_model)
            if lower_rchisqr < self._shown_rchisqr :
                self._shown_model = upper_optimizer.shown_model
                self._shown_covariance = upper_optimizer.shown_covariance
                self._shown_rchisqr = upper_rchisqr

        return self.shown_model, self.shown_covariance

    # changes top5
    # changes _shown variables
    def find_best_model_for_dataset(self, halved=0):

        if not halved :
            self.build_composite_function_list()

        x_points, y_points, sigma_points, use_errors = self.fit_setup()

        num_models = len(self._composite_function_list)
        for idx, model in enumerate(self._composite_function_list) :

            fitted_model, fitted_cov = self.fit_loop(model_=model,
                                                     x_points=x_points, y_points=y_points, sigma_points=sigma_points,
                                                     use_errors=use_errors,info_string=f"{idx+1}/{num_models} ")

            self.query_add_to_top5(model=fitted_model, covariance=fitted_cov)




        print(f"\nBest models are {[m.name for m in self.top5_models]} with "
              f"associated reduced chi-squareds {self.top5_rchisqrs}")

        print(f"\nBest model is {self.top_model} "
              f"\n with args {self.top_args} += {self.top_uncs} "
              f"\n and reduced chi-sqr {self.top_rchisqr}")
        self.top_model.print_tree()

        if self.top_rchisqr > 5 and len(self._data) > 20:
            # if a better model is found here it will probably underestimate the actual rchisqr since
            # it will only calculate based on half the data
            print("It is unlikely that we have found the correct model... halving the dataset")
            lower_data = self._data[:math.floor(len(self._data)/2)]
            lower_optimizer = Optimizer(data=lower_data,
                                        use_functions_dict=self._use_functions_dict,
                                        max_functions=self._max_functions)
            lower_optimizer.composite_function_list = self._composite_function_list
            lower_optimizer.find_best_model_for_dataset()
            for l_model, l_cov in zip(lower_optimizer.top5_models,lower_optimizer.top5_covariances) :
                self.query_add_to_top5(model=l_model,covariance=l_cov)

            upper_data = self._data[math.floor(len(self._data)/2):]
            upper_optimizer = Optimizer(data=upper_data,
                                        use_functions_dict=self._use_functions_dict,
                                        max_functions=self._max_functions)
            upper_optimizer.composite_function_list = self._composite_function_list
            upper_optimizer.find_best_model_for_dataset()
            for u_model, u_cov in zip(upper_optimizer.top5_models,upper_optimizer.top5_covariances) :
                self.query_add_to_top5(model=u_model,covariance=u_cov)
    # changes top5
    # changes _shown variables
    def async_find_best_model_for_dataset(self, start=False):

        done_flag = 0
        if start:
            self._composite_generator = self.all_valid_composites_generator()

        x_points, y_points, sigma_points, use_errors = self.fit_setup(fourier_condition=start)

        batch_size = 10
        for idx in range(batch_size) :

            try:
                model = next(self._composite_generator)
            except StopIteration :
                done_flag = 1
                break

            fitted_model, fitted_cov = self.fit_loop(model_=model,
                                                     x_points=x_points, y_points=y_points, sigma_points=sigma_points,
                                                     use_errors=use_errors, info_string=f"Async {idx}/{self._gen_idx+1} ")

            self.query_add_to_top5(model=fitted_model, covariance=fitted_cov)


        self.shown_model = self.top5_models[0]
        self.shown_covariance = self.top5_covariances[0]
        self.shown_rchisqr = self.top5_rchisqrs[0]

        if done_flag :
            return "Done"


    def create_cos_sin_frequency_lists(self):

        self._cos_freq_list = []
        self._sin_freq_list = []

        # need to recreate the data as a list of uniform x-interval pairs
        # (in case the data points aren't uniformly spaced)
        x_points = np.linspace( min( [datum.pos for datum in self._data] ),
                                max( [datum.pos for datum in self._data] ),
                                num = len(self._data) + 1  # supersample of the data, with endpoints
                                                           # the same and one more for each interval
                              )
        # very inefficient
        y_points = [ self._data[0].val ]
        for target in x_points[1:-1] :
            # find the two x-values in _data that surround the target value
            for idx, pos_high in enumerate( [datum.pos for datum in self._data] ):
                if pos_high > target :
                    # equation of the line connecting the two points is
                    # y(x) = [ y_low(x_high-x) + y_high(x-x_low) ] / (x_high - x_low)
                    x_low, y_low = self._data[idx-1].pos, self._data[idx-1].val
                    x_high, y_high = self._data[idx].pos, self._data[idx].pos
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

        pos_Ynu = fft_Ynu[ math.floor(len(fft_Ynu)/2) : ]  # the positive frequency values
        pos_nu = fft_nu[ math.floor(len(fft_Ynu)/2) : ]    # the positive frequencies

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
            test_func = CompositeFunction(prim_=PrimitiveFunction.built_in("sin"),
                                          children_list=[PrimitiveFunction.built_in("pow1")])
            sin_rchisqr_list = []
            for i in range(7) :
                test_func.set_args(2*np.abs(pos_Ynu[argmax]), 2*math.pi*(pos_nu[argmax]+delta_nu*(i-3)/3) )
                sin_rchisqr_list.append( self.reduced_chi_squared_of_fit(model=test_func) )

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
            test_func = CompositeFunction(prim_=PrimitiveFunction.built_in("cos"),
                                          children_list=[PrimitiveFunction.built_in("pow1")])
            cos_rchisqr_list = []
            for i in range(7):
                test_func.set_args( 2*np.abs(pos_Ynu[argmax]), 2*math.pi*(pos_nu[argmax] + delta_nu*(i-3) / 3) )
                cos_rchisqr_list.append(self.reduced_chi_squared_of_fit(model=test_func))

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
                # print(f"Adding {best_freq} to sin")
                self._sin_freq_list.append(best_freq)
            else:
                # print(f"Adding {best_freq} to cos")
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
            # print(f"Using weighted {best_rchisqr=} {best_grid_point}")

        model.set_args( *best_grid_point )
        # model.print_tree()

        return best_grid_point
    def find_set_initial_guess_scaling(self, composite: CompositeFunction):

        for child in composite.children_list :
            self.find_set_initial_guess_scaling(child)

        # use knowledge of scaling to guess parameter sizes from the characteristic sizes in the data
        charAvY = (max([datum.val for datum in self._data]) + min([datum.val for datum in self._data])) / 2
        charDiffY = (max([datum.val for datum in self._data]) - min([datum.val for datum in self._data])) / 2
        charAvX = ( max( [datum.pos for datum in self._data] ) + min( [datum.pos for datum in self._data] ) ) / 2
        charDiffX =  (max([datum.pos for datum in self._data]) - min([datum.pos for datum in self._data])) / 2
        if composite.prim.name == "pow0" :
            # this typically represents a shift, so the average X is more important than the range of x-values
            charX = charAvX
        else :
            charX = charDiffX

        # defaults
        xmul = charX ** composite.dimension_arg
        ymul = 1

        # overrides
        if composite.parent is None :
            if composite.prim.name[:2] != "n_" and composite.prim.name != "sum_" :
                ymul = charDiffY
            else:
                pass
        elif composite.parent.prim.name == "sum_" and composite.parent.parent is None:
            ymul = charDiffY
        elif composite.parent.prim.name == "my_cos" :
            # in cos( Aexp(Lx) ), for small x the inner composition goes like A + ALx
            # = f(0) + x f'(0) and here f'(0) should correspond to the largest fourier component
            # problem is that this answer should be independent of the initial parameter A... it only works from scratch
            # since we set A = 1 for new composites
            slope_at_zero = (composite.eval_at(2e-5)-composite.eval_at(1e-5) ) / (1e-5 * composite.prim.arg)
            if abs(slope_at_zero) > 1e-5 :
                if len(self._cos_freq_list_dup) > 0 :
                    # print(f"Using cosine frequency {2*math.pi*self._cos_freq_list_dup[0]}")
                    xmul = ( 2*math.pi*self._cos_freq_list_dup.pop(0) ) / slope_at_zero
                else:  # misassigned cosine frequency
                    try:
                        xmul = ( 2*math.pi*self._sin_freq_list_dup.pop(0) ) / slope_at_zero
                    except TypeError :
                        print(self._sin_freq_list_dup)
                        print(self._sin_freq_list)
                        print(self._cos_freq_list_dup)
                        print(self._cos_freq_list)
                        raise SystemExit
        elif composite.parent.prim.name == "my_sin" :
            slope_at_zero = (composite.eval_at(2e-5)-composite.eval_at(1e-5) ) / (1e-5 * composite.prim.arg)
            if abs(slope_at_zero) > 1e-5 :
                if len(self._sin_freq_list_dup) > 0 :
                    # print(f"Using sine frequency {2*math.pi*self._sin_freq_list_dup[0]}")
                    xmul = ( 2*math.pi*self._sin_freq_list_dup.pop(0) ) / slope_at_zero
                else:  # misassigned cosine frequency
                    xmul = ( 2*math.pi*self._cos_freq_list_dup.pop(0) ) / slope_at_zero

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
        # print("Optimizer show fit: here once")
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
    def modified_chi_sqr_for_believability(self, model):
        avg_grid_spacing = (self._data[-1].pos - self._data[0].pos) / len(self._data)
        for idx in range(model.num_nodes()):
            node = model.get_node_with_index(idx)
            if node.parent is not None and node.parent.prim.name in ["my_sin", "my_cos"] :
                if node.prim.arg > 10*avg_grid_spacing :
                    print("LIAR! FREQUENCY EXTRAVAGENT")
                    return self.reduced_chi_squared_of_fit(model) * (node.prim.arg/avg_grid_spacing)**2
        return self.reduced_chi_squared_of_fit(model)

    def r_squared(self, model):
        mean = sum( [datum.val for datum in self._data] )/len(self._data)
        variance_data = sum( [ (datum.val-mean)**2 for datum in self._data ] )/len(self._data)
        variance_fit = sum( [ (datum.val-model.eval_at(datum.pos))**2 for datum in self._data ] )/len(self._data)
        return 1 - variance_fit/variance_data

    def smoothed_data(self, n=1):

        print("Using smoothed data")
        raise RuntimeError

        data_to_smooth = []
        return_data = []
        if n <= 1 :
            data_to_smooth = self._averaged_data
        else :
            data_to_smooth = self.smoothed_data(n-1)
        for idx, datum in enumerate(data_to_smooth[:-1]) :
            new_pos = (data_to_smooth[idx+1].pos + data_to_smooth[idx].pos) / 2
            new_val = (data_to_smooth[idx+1].val + data_to_smooth[idx].val) / 2
            # propagation of uncertainty
            new_sigma_pos = math.sqrt( (data_to_smooth[idx+1].sigma_pos/2)**2
                                       + (data_to_smooth[idx].sigma_pos/2)**2 )
            new_sigma_val = math.sqrt( (data_to_smooth[idx+1].sigma_val/2)**2
                                       + (data_to_smooth[idx].sigma_val/2)**2 )
            return_data.append( Datum1D(pos= new_pos, val=new_val, sigma_pos=new_sigma_pos, sigma_val=new_sigma_val) )
        return return_data
    def deriv_on_n_smoothed(self, n=1):

        # this assumes that data is sequential, i.e. that there are no repeated measurements for each x position

        data_to_deriv = None
        return_deriv = []
        if n <= 0 :
            data_to_deriv = self._averaged_data
        else :
            data_to_deriv = self.smoothed_data(n-1)
        for idx, datum in enumerate(data_to_deriv[:-2]) :
            new_deriv = ( (data_to_deriv[idx+2].val - data_to_deriv[idx].val)
                             / (data_to_deriv[idx+2].pos - data_to_deriv[idx].pos) )
            new_pos = (data_to_deriv[idx+2].val - data_to_deriv[idx].val) / 2

            # propagation of uncertainty
            new_sigma_pos = math.sqrt( (data_to_deriv[idx+2].sigma_pos/2)**2
                                       + (data_to_deriv[idx].sigma_pos/2)**2 )
            new_sigma_deriv = ( (1/(data_to_deriv[idx+2].pos-data_to_deriv[idx].pos) ) *
                                math.sqrt( data_to_deriv[idx+2].sigma_val**2 + data_to_deriv[idx].sigma_val**2 )
                                + new_deriv**2 * ( data_to_deriv[idx+2].sigma_pos**2 + data_to_deriv[idx].sigma_pos**2 )
                            )
            return_deriv.append(Datum1D(pos=new_pos, val=new_deriv, sigma_pos=new_sigma_pos, sigma_val=new_sigma_deriv))

        return return_deriv
    def average_data(self):
        # this is only used to find an initial guess for the data, and so more sophisticated techniques like
        # weighted means is not reqired

        # get means and number of pos-instances into dict
        sum_val_dict = defaultdict(float)
        num_dict = defaultdict(int)
        for datum in self._data :
            if sum_val_dict[datum.pos] :
                sum_val_dict[datum.pos] += datum.val
                num_dict[datum.pos] += 1
            else :
                sum_val_dict[datum.pos] = datum.val
                num_dict[datum.pos] = 1

        mean_dict = {}
        for ikey, isum in sum_val_dict.items() :
            mean_dict[ikey] = isum / num_dict[ikey]

        propagation_variance_x_dict = defaultdict(float)
        propagation_variance_y_dict = defaultdict(float)
        sample_variance_y_dict = defaultdict(float)
        for datum in self._data :
            # average the variances
            propagation_variance_x_dict[datum.pos] += datum.sigma_pos**2
            propagation_variance_y_dict[datum.pos] += datum.sigma_val**2
            sample_variance_y_dict[datum.pos] += (datum.val-mean_dict[datum.pos])**2

        averaged_data = []
        for key, val in mean_dict.items() :


            sample_uncertainty_squared = sample_variance_y_dict[key] / (num_dict[key]-1) if num_dict[key] > 1 else 0
            propagation_uncertainty_squared = propagation_variance_y_dict[key] / num_dict[key]
            ratio = ( sample_uncertainty_squared / (sample_uncertainty_squared + propagation_uncertainty_squared)
                      if propagation_uncertainty_squared > 0 else 1 )

            # interpolates smoothly between 0 uncertainty in data points (so all uncertainty comes from sample spread)
            # to the usual uncertainty coming from both the data and the spread
            # TODO: see
            #  https://stats.stackexchange.com/questions/454120/how-can-i-calculate-uncertainty-of-the-mean-of-a-set-of-samples-with-different-u
            # for a more rigorous treatment
            effective_uncertainty_squared = ratio*sample_uncertainty_squared + (1-ratio)*propagation_uncertainty_squared

            averaged_data.append( Datum1D(pos=key, val=val,
                                          sigma_pos=math.sqrt( propagation_variance_x_dict[key] ),
                                          sigma_val=math.sqrt( effective_uncertainty_squared )
                                         )
                                )
        self._averaged_data = sorted(averaged_data)


    def load_default_functions(self):
        self._primitive_function_list.extend( [ PrimitiveFunction.built_in("pow0"),
                                                PrimitiveFunction.built_in("pow1") ] )
    def load_non_defaults_from(self,use_functions_dict) :
        for key, use_function in use_functions_dict.items():
            if key == "cos(x)" and use_function:
                self.load_cos_function()
            if key == "sin(x)" and use_function:
                self.load_sin_function()
            if key == "exp(x)" and use_function:
                self.load_exp_function()
            if key == "log(x)" and use_function:
                self.load_log_function()
            if key == "1/x" and use_function:
                self.load_pow_neg1_function()
            if key == "x\U000000B2" and use_function:
                self.load_pow2_function()
            if key == "x\U000000B3" and use_function:
                self.load_pow3_function()
            if key == "x\U00002074" and use_function:
                self.load_pow4_function()
            if key == "custom" and use_function:
                self.load_custom_functions()

    def load_cos_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("cos") )
    def unload_cos_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("cos") )
    def load_sin_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("sin") )
    def unload_sin_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("sin") )

    def load_exp_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("exp") )
    def unload_exp_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("exp") )
    def load_log_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("log" ) )
    def unload_log_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("log") )

    def load_pow_neg1_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("pow_neg1") )
    def unload_pow_neg1_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("pow_neg1") )
    def load_pow2_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("pow2") )
    def unload_pow2_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("pow2") )
    def load_pow3_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("pow3") )
    def unload_pow3_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("pow3") )
    def load_pow4_function(self):
        self._primitive_function_list.append( PrimitiveFunction.built_in("pow4") )
    def unload_pow4_function(self):
        self._primitive_function_list.remove( PrimitiveFunction.built_in("pow4") )

    def load_custom_functions(self):
        pass
    def unload_custom_functions(self):
        pass


def std(cov):
    return list(np.sqrt(np.diagonal(cov)))

def list_sums_weights(l1,l2,w1,w2) -> list[float]:
    if len(l1) != len(l2) :
        return []
    sum_list = []
    for val1, val2 in zip(l1,l2) :
        sum_list.append(val1*w1+val2*w2)
    return sum_list
