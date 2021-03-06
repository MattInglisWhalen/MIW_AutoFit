
# default libraries
import math
import re as regex
from collections import defaultdict
import codeop

# external libraries
import numpy
import scipy.optimize
from scipy.optimize import curve_fit, differential_evolution
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


class Optimizer:

    def __init__(self, data=None, use_functions_dict = None, max_functions=3):

        print("New optimizer created")

        # datasets, which are lists of Datum1D instances
        self._data = []             # the raw datapoints of (x,y), possibly with uncertainties
                                    # May also represent a binned point if the input file is a list of x_values only
        self._averaged_data = []    # the model optimizer requires approximate slope knowledge,
                                    # so multiple measurements at each x-value are collapsed into a single value

        # fit results
        self._best_function = None          # a CompositeFunction
        self._best_args = None              # the arguments (parameters) of best_function CompositeFunction
        self._best_args_uncertainty = None  # the uncorrelated uncertainties of the arguments
        self._best_red_chi_sqr = 1e5

        # top 5 models and their data
        self._top_5_models = []
        self._top_5_args = []
        self._top_5_uncertainties = []
        self._top_5_names = [""]
        self._top_5_red_chi_squareds = [1e5 for _ in range(5)]

        # function construction parameters
        self._max_functions = max_functions
        self._primitive_function_list = []
        self._primitive_names_list = []

        # useful auxiliary varibles
        self._temp_function = None      # a CompositeFunction
        self._cos_freq_list = None
        self._sin_freq_list = None
        self._all_freq_list = None
        self._cos_freq_list_dup = None
        self._sin_freq_list_dup = None
        self._all_freq_list_dup = None

        if data is not None :
            self._data = sorted(data)  # list of Datum1D. Sort makes the data monotone increasing in x-value
            self.average_data()

        self.load_default_functions()
        self.load_non_defaults_from(use_functions_dict)

        self._composite_function_list = []
        self._composite_generator = None
        self._gen_idx = -1


    def __repr__(self):
        pass

    @property
    def best_model(self):
        return self._best_function
    @property
    def parameters(self):
        return self._best_args
    @parameters.setter
    def parameters(self, vals_list):
        self._best_args = vals_list
        self._best_function.set_args(*vals_list)
    @property
    def uncertainties(self):
        return self._best_args_uncertainty
    @uncertainties.setter
    def uncertainties(self, uncs_list):
        self._best_args_uncertainty = uncs_list
    @property
    def top5_models(self):
        return self._top_5_models
    @property
    def top5_args(self):
        return self._top_5_args
    @property
    def top5_uncertainties(self):
        return self._top_5_uncertainties
    @property
    def top5_names(self):
        return self._top_5_names
    @property
    def top5_rx_sqrs(self):
        return self._top_5_red_chi_squareds
    def query_add_to_top5(self, model, pars, uncertainties, red_chi_squared):
        if len(self._top_5_models) > 5 and red_chi_squared > self._top_5_red_chi_squareds[-1] :
            return
        if model.name in self.top5_names :
            return
        for idx, chi_sqr in enumerate(self._top_5_red_chi_squareds) :
            if self.reduced_chi_squared_of_fit(model) < self._top_5_red_chi_squareds[idx] :
                self._top_5_models.insert(idx, model.copy())
                self._top_5_args.insert(idx, pars)
                self._top_5_uncertainties.insert(idx, uncertainties)
                self._top_5_names.insert(idx, model.name)
                self._top_5_red_chi_squareds.insert(idx, red_chi_squared)

                self._top_5_models = self._top_5_models[:5]
                self._top_5_args = self._top_5_args[:5]
                self._top_5_uncertainties = self._top_5_uncertainties[:5]
                self._top_5_names = self._top_5_names[:5]
                self._top_5_red_chi_squareds = self._top_5_red_chi_squareds[:5]

                print(f"New top {idx} with red_chisqr={red_chi_squared:.2F}: {model=} with pars={pars} \u00B1 {uncertainties}")
                model.print_tree()
                return

    def models_left(self):
        return len(self._composite_function_list)

    def composite_function_generator(self, depth):

        self._primitive_names_list = [iprim.name for iprim in self._primitive_function_list]

        if depth <= 1 :
            for iprim in self._primitive_function_list:
                new_comp = CompositeFunction(func=iprim)
                yield new_comp

        else :
            head_gen = self.composite_function_generator( depth = depth-1 )
            for icomp in head_gen :
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list:
                        new_comp = icomp.copy()
                        new_comp.get_node_with_index(idescendent).add_child(iprim)
                        new_comp.build_name()
                        yield new_comp

    def valid_composite_function_generator(self, depth):

        all_comps_at_depth = self.composite_function_generator(depth)
        for icomp in all_comps_at_depth :
            if self.passes_rules(icomp):
                yield icomp

    def all_valid_composites_generator(self):
        for idepth in range(7):
            for icomp in self.valid_composite_function_generator(depth=idepth):
                self._gen_idx += 1
                yield icomp

    # TODO: make a generator version of this e.g. [] -> ()
    def build_composite_function_list(self):

        # self.add_primitive_to_list(name="my_tanh",functional_form="np.tanh(x)")
        # self.add_primitive_to_list(name="my_tanh", functional_form="x if 0>1 else PrimitiveFunction._built_in_prims_dict.clear()")

        # start with simple primitives
        self._composite_function_list = []
        for iprim in self._primitive_function_list:
            new_comp = CompositeFunction(func=iprim)
            self._composite_function_list.append(new_comp)

        for _ in range(self._max_functions-1) :
            new_list = self._composite_function_list.copy()
            for icomp in self._composite_function_list[:]:
                for idescendent in range(icomp.num_nodes()):
                    for iprim in self._primitive_function_list:
                        new_comp = icomp.copy()
                        new_comp.get_node_with_index(idescendent).add_child(iprim)
                        new_comp.build_name()
                        new_list.append(new_comp)
            self._composite_function_list = new_list.copy()

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

            if not self.passes_rules(icomp) :
                self._composite_function_list.remove(icomp)

        # remove duplicates using a dict{}
        self._composite_function_list = list(
            { icomp.__repr__() : icomp for icomp in self._composite_function_list[:] }.values()
        )

    def passes_rules(self, icomp):

        remove_flag = 0
        name = icomp.name

        # composition of a constant function is wrong
        if regex.search(f"pow0\(", name):
            remove_flag = 2
        if regex.search(f"\(pow0\)", name):
            remove_flag = 3

        # composition of powers with no sum is wrong
        if regex.search(f"pow[0-9]\(pow[a-z0-9_]*\)", name):
            remove_flag = 5

        # repeated reciprocal is wrong
        if regex.search(f"pow_neg1\(pow_neg1\)", name):
            remove_flag = 7
            # and deeper
        if regex.search(f"pow_neg1\(pow_neg1\([a-z0-9_+]*]\)\)", name):
            remove_flag = 7

        # trivial reciprocal is wrong
        if regex.search(f"pow_neg1\(pow1\)", name):
            remove_flag = 11

        # composition of pow1 with no sum is wrong
        if regex.search(f"pow1\([a-z0-9_]*\)", name):
            remove_flag = 13

        # deeper composition of pow1 with no sum is wrong
        for prim_name1 in self._primitive_names_list:
            for prim_name2 in self._primitive_names_list:
                if regex.search(f"pow1\({prim_name1}\({prim_name2}\)\)", name):
                    remove_flag = 17

        # sum of pow1 with composition following is wrong
        if regex.search(f"\+pow1\(", name):
            remove_flag = 19

        # sum of the same function is wrong
        for prim_name in self._primitive_names_list:
            if regex.search(f"{prim_name}[a-z0-9_+]*{prim_name}", name):
                remove_flag = 23

        # trig inside trig is too complex, and very (very) rarely occurs in applications
        if icomp.has_double_trigness():
            remove_flag = 29
        if icomp.has_double_expness():
            remove_flag = 29
        if icomp.has_double_logness():
            remove_flag = 29

        if name[0:4] == "pow1" and icomp.num_children() == 1:
            remove_flag = 31

        # pow0+log(...) is a duplicate since A + Blog( Cf(x) ) = B log( exp(A/B) Cf(x) ) = log(f)
        if regex.search(f"pow0\+my_log\([a-z0-9_]*\)", name) \
                or regex.search(f"my_log\([a-z0-9_]*\)\+pow0", name):
            remove_flag = 37

        # sins and cosines and exps and logs never have angular frequency or decay parameters exactly 1
        # all unitless-argument functions start with my_
        if regex.search(f"my_[a-z]*\+", name) or regex.search(f"my_[a-z]*\)", name):
            if regex.search(f"my_exp\(my_log\)", name):
                # the one exception in order to get power laws
                pass
            else:
                remove_flag = 43

        # trivial composition should just be straight sums
        if regex.search(f"\(pow1\(", name) or regex.search(f"\+pow1\(", name):
            remove_flag = 47

        # more more exp algebra: Alog( Bexp(Cx) ) = AlogB + ACx = pow0+pow1
        if regex.search(f"my_log\(my_exp\(pow1\)\)", name):
            remove_flag = 53

        # reciprocal exponent is already an exponent pow_neg1(exp) == my_exp(pow1)
        if regex.search(f"pow_neg1\(my_exp\)", name):
            remove_flag = 61
        # and again but deeper
        if regex.search(f"pow_neg1\(my_exp\([a-z0-9_+]*\)\)", name):
            remove_flag = 61

        # log of "higher" powers is the same as log(pow1)
        if regex.search(f"my_log\(pow[a-z2-9_]*\)", name) or regex.search(f"my_log\(pow_neg1\)", name):
            remove_flag = 67

        # exp(pow0+...) is just exp(...)
        if regex.search(f"my_exp\(pow0", name) or regex.search(f"my_exp\([a-z0-9_+]*pow0", name):
            remove_flag = 71

        # log(1/f+g) of log( (f+g)^n )is the same as log(f+g)
        if regex.search(f"my_log\(pow[a-z0-9_]*\([a-z0-9_+]*\)\)", name):
            remove_flag = 73

        # pow3(exp(...)) is just exp(3...)
        if regex.search(f"pow[a-z0-9_]*\(my_exp\([a-z0-9_+]*\)\)", name):
            remove_flag = 83

        if name == "my_exp(my_log)":
            if remove_flag:
                print(f"\n\n>>> Why did we remove {name=} at {remove_flag=} <<<\n\n")
                raise SystemExit

        if name == "pow1(my_cos(pow1)+my_sin(pow1))":
            if remove_flag:
                print(f"Why did we remove {name=} at {remove_flag=}")
                raise SystemExit

        if name == "pow_neg1":
            if remove_flag:
                print(f"Why did we remove {name=} with {remove_flag=}")
                raise SystemExit

        if name == "pow1(pow0+pow0)":
            if not remove_flag:
                print(f"Why didn't we remove {name=} with {remove_flag=}")
                raise SystemExit

        if remove_flag :
            return False
        # else :
        return True

    def add_primitive_to_list(self, name, functional_form):
        # For adding one's own Primitive Functions to the built-in list

        print(f"\n{name} {functional_form}")
        if regex.search("\\\\",functional_form) or regex.search("\n",functional_form)\
                or regex.search("\s",functional_form) :
            print("Stop trying to inject code")
            return

        code_str  = f"def {name}(x,arg):\n"
        code_str += f"    return arg*{functional_form}\n"
        code_str += f"new_prim = PrimitiveFunction(func={name})\n"
        code_str += f"dict = PrimitiveFunction.built_in_dict()\n"
        code_str += f"dict[\"{name}\"] = new_prim\n"

        print(f">{code_str}<")
        exec(code_str)

        for key, item in PrimitiveFunction.built_in_dict().items() :
            print(f"{key}, {item}, {item.eval_at(1.5)}")

        self._primitive_function_list.append( PrimitiveFunction.built_in(name) )

        pass

    def set_data_to(self, other_data):
        self._data = sorted(other_data)
        self.average_data()

    def parameters_and_uncertainties_from_fitting(self, model = None, initial_guess = None):

        if model is None:
            model = self._best_function.copy()
            model.print_tree()

        x_points = []
        y_points = []
        sigma_points = []

        use_errors = True

        for datum in self._data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            if datum.sigma_val < 1e-5:
                print(" >>> NO ERROR <<< ")
                use_errors = False
                sigma_points.append(1.)
            else:
                sigma_points.append(datum.sigma_val)

        if model.num_trig() > 0:
            self.create_cos_sin_frequency_lists()

        # Find an initial guess for the parameters based off scaling arguments
        # The loss function there also tries to fit the data's smoothed derivatives
        if initial_guess is None :
            initial_guess = self.find_initial_guess_scaling(model)
            model.set_args(*initial_guess)

        # Next, find a better guess by relaxing the error bars on the data
        # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
        # and so relaxed error bars help point towards global minima
        try:
            better_guess, better_unc = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                                 p0=initial_guess, maxfev=5000)
        except RuntimeError:
            print("Couldn't find optimal parameters for a better guess.")
            self._best_args = initial_guess
            self._best_args_uncertainty = [1e5 for _ in range(model.dof)]
            return self._best_args, self._best_args_uncertainty

        # Finally, use the better guess to find the true minimum with the true error bars
        try:
            np_pars, np_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                        sigma=sigma_points, absolute_sigma=use_errors,
                                        p0=better_guess, maxfev=5000)
        except RuntimeError:
            print("Couldn't find optimal parameters for the final fit. ")
            self._best_args = better_guess
            self._best_args_uncertainty = better_unc
            return self._best_args, self._best_args_uncertainty

        pars = np_pars.tolist()
        uncertainties = np.sqrt(np.diagonal(np_cov)).tolist()
        model.set_args(*pars)  # ignore iterable

        # TODO: best_function is either outdated or nees a name change
        self._best_function = model.copy()
        self._best_args = pars  # have to do this because the model's stored parameters change for some reason
        self._best_args_uncertainty = uncertainties

        return pars, uncertainties

    def find_best_model_for_dataset(self):

        self.build_composite_function_list()
        self.create_cos_sin_frequency_lists()
        # self._top_5_red_chi_squareds = [1e5 for _ in range(5)]

        x_points = []
        y_points = []
        sigma_points = []

        use_errors = True

        for datum in self._data :
            x_points.append( datum.pos )
            y_points.append( datum.val )
            if datum.sigma_val < 1e-5 :
                use_errors = False
                sigma_points.append( 1. )
                datum.sigma_val = 1
            else:
                sigma_points.append( datum.sigma_val )

        num_models = len(self._composite_function_list)
        for idx, model in enumerate(self._composite_function_list) :

            print(f"\nFitting {model=}")
            print(model.tree_as_string_with_dimensions())

            # do an FFT if there's a sin/cosine --
            # we zero-out the average height of the data to remove the 0-frequency mode as a contribution
            # then to pass in the dominant frequencies as arguments to the initial guess
            if model.num_trig() > 0 :
                self._all_freq_list_dup = self._all_freq_list.copy()
                self._cos_freq_list_dup = self._cos_freq_list.copy()
                self._sin_freq_list_dup = self._sin_freq_list.copy()

            # Find an initial guess for the parameters based off of scaling arguments
            initial_guess = self.find_initial_guess_scaling(model)
            model.set_args(*initial_guess)
            print(f"{idx+1}/{num_models} Scaling guess: {initial_guess}")

            # Next, find a better guess by relaxing the error bars on the data
            # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
            # and so relaxed error bars help point towards global minima
            try:
                better_guess, _ = curve_fit( model.scipy_func, xdata=x_points, ydata=y_points,
                                             p0=initial_guess, maxfev=5000, method='lm')
            except RuntimeError :
                print("Couldn't find optimal parameters. Continuing on...")
                continue
            print(f"{idx+1}/{num_models} Better guess: {better_guess}")

            # Finally, use the better guess to find the true minimum with the true error bars
            try:
                np_pars, np_cov = curve_fit( model.scipy_func, xdata=x_points, ydata=y_points,
                                             sigma=sigma_points, absolute_sigma=use_errors,
                                             p0=better_guess, maxfev=5000)
            except RuntimeError :
                print("Couldn't find optimal parameters. Continuing on...")
                continue
            print(f"{idx+1}/{num_models} Final guess: {np_pars}")

            pars = np_pars.tolist()
            uncertainties = np.sqrt(np.diagonal(np_cov)).tolist()
            model.set_args( *pars )     # ignore iterable
            # could also try setting uncertainties, and you'd get a relatively simple way to find error bands

            # goodness of fit: reduced chi_R^2 = chi^2 / (N-k) should be close to 1
            red_chisqr = self.reduced_chi_squared_of_fit(model, use_errors)

            self.query_add_to_top5(model=model, pars=pars, uncertainties=uncertainties, red_chi_squared=red_chisqr)


        print(self._top_5_models)
        print(self._top_5_red_chi_squareds)

        self._best_function = self._top_5_models[0]
        self._best_args = self._top_5_args[0]
        self._best_args_uncertainty = self._top_5_uncertainties[0]
        self._best_red_chi_sqr = self._top_5_red_chi_squareds[0]
        print(f"\nBest model is {self._best_function} "
              f"\n with args {self._best_args} += {self._best_args_uncertainty} "
              f"\n and reduced chi-sqr {self._best_red_chi_sqr}")
        self._best_function.set_args( *self._best_args )  # ignore iterable
        self._best_function.print_tree()

    def async_find_best_model_for_dataset(self, start=False):

        x_points = []
        y_points = []
        sigma_points = []

        use_errors = True
        done_flag = 0

        for datum in self._data :
            x_points.append( datum.pos )
            y_points.append( datum.val )
            if datum.sigma_val < 1e-5 :
                use_errors = False
                sigma_points.append( 1. )
                datum.sigma_val = 1
            else:
                sigma_points.append( datum.sigma_val )

        if start :
            self._composite_generator = self.all_valid_composites_generator()
            self.create_cos_sin_frequency_lists()

        batch_size = 10

        for idx in range(batch_size) :

            try:
                model = next(self._composite_generator)
            except StopIteration :
                done_flag = 1
                break

            print(f"\nFitting {model=}")
            print(model.tree_as_string_with_dimensions())

            # do an FFT if there's a sin/cosine --
            # we zero-out the average height of the data to remove the 0-frequency mode as a contribution
            # then to pass in the dominant frequencies as arguments to the initial guess
            if model.num_trig() > 0:
                self._all_freq_list_dup = self._all_freq_list.copy()
                self._cos_freq_list_dup = self._cos_freq_list.copy()
                self._sin_freq_list_dup = self._sin_freq_list.copy()

            # Find an initial guess for the parameters based off of scaling arguments
            initial_guess = self.find_initial_guess_scaling(model)
            model.set_args(*initial_guess)
            print(f"Async {self._gen_idx+1} -- Scaling guess: {initial_guess}")

            # Next, find a better guess by relaxing the error bars on the data
            # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
            # and so relaxed error bars help point towards global minima
            try:
                better_guess, _ = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                            p0=initial_guess, maxfev=5000, method='lm')
            except RuntimeError:
                print("Couldn't find optimal parameters. Continuing on...")
                continue
            print(f"Async {self._gen_idx+1} -- Better guess: {better_guess}")

            # Finally, use the better guess to find the true minimum with the true error bars
            try:
                np_pars, np_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                            sigma=sigma_points, absolute_sigma=use_errors,
                                            p0=better_guess, maxfev=5000)
            except RuntimeError:
                print("Couldn't find optimal parameters. Continuing on...")
                continue
            print(f"Async {self._gen_idx+1} -- Final guess: {np_pars}")

            pars = np_pars.tolist()
            uncertainties = np.sqrt(np.diagonal(np_cov)).tolist()
            model.set_args(*pars)  # ignore iterable
            # could also try setting uncertainties, and you'd get a relatively simple way to find error bands

            # goodness of fit: reduced chi_R^2 = chi^2 / (N-k) should be close to 1
            red_chisqr = self.reduced_chi_squared_of_fit(model, use_errors)

            self.query_add_to_top5(model=model, pars=pars, uncertainties=uncertainties, red_chi_squared=red_chisqr)

        self._best_function = self._top_5_models[0]
        self._best_args = self._top_5_args[0]
        self._best_args_uncertainty = self._top_5_uncertainties[0]
        self._best_red_chi_sqr = self._top_5_red_chi_squareds[0]
        print(f"\nBest model is {self._best_function} "
              f"\n with args {self._best_args} += {self._best_args_uncertainty} "
              f"\n and reduced chi-sqr {self._best_red_chi_sqr}")
        self._best_function.set_args(*self._best_args)  # ignore iterable
        self._best_function.print_tree()

        if done_flag :
            return "Done"


    def create_cos_sin_frequency_lists(self):

        self._all_freq_list = []
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

        avg_y = sum(y_points) / len(y_points)
        zeroed_y_points = [val - avg_y for val in y_points]

        # complex fourier spectrum, with positions adjusted to frequency space
        fft_Ynu = fftshift(fft(zeroed_y_points)) / len(zeroed_y_points)
        fft_nu = fftshift(fftfreq(len(zeroed_y_points), x_points[1] - x_points[0]))

        pos_Ynu = fft_Ynu[ math.floor(len(fft_Ynu)/2) : ]  # the positive frequency values
        pos_nu = fft_nu[ math.floor(len(fft_Ynu)/2) : ]    # the positive frequencies

        for n in range( len(pos_nu) ):
            argmax = np.argmax([np.abs(Ynu) for Ynu in pos_Ynu])
            if np.abs(np.real(pos_Ynu[argmax])) > np.abs(np.imag(pos_Ynu[argmax])) :
                self._cos_freq_list.append( pos_nu[argmax] )
            else:
                self._sin_freq_list.append(pos_nu[argmax])
            self._all_freq_list.append( pos_nu[argmax] )
            pos_Ynu = np.delete(pos_Ynu, argmax)
            pos_nu = np.delete(pos_nu, argmax)


    # def find_initial_guess(self, model: CompositeFunction, width = 10., num_each=10):
    #
    #     best_rchisqr = 1e5
    #     best_grid_point = model.get_args()
    #
    #     total_grid = np.mgrid[tuple(slice(-width / 2, width / 2, complex(0, num_each)) for _ in range(model.dof))]
    #
    #     args = []
    #     for idim in range( model.dof ):
    #         args.append(total_grid[idim].flatten())
    #     for ipoint in zip(*args):
    #         model.set_args( *ipoint )
    #         temp_rchisqr = self.reduced_chi_squared_of_fit(model)
    #         if temp_rchisqr < best_rchisqr :
    #             best_rchisqr = temp_rchisqr
    #             best_grid_point = list(ipoint)
    #
    #     model.set_args( *best_grid_point)
    #
    #     return best_grid_point

    # def find_initial_guess_genetic(self, model: CompositeFunction):
    #
    #     max_Y_data = max( [datum.val for datum in self._data] ) - min( [datum.val for datum in self._data] )
    #     max_X_data = max( [datum.pos for datum in self._data] ) - min( [datum.pos for datum in self._data] )
    #     max_data = max( max_X_data, max_Y_data )
    #
    #     parameter_bounds = []
    #     for _ in range(model.dof):
    #         parameter_bounds.append( [-max_data, max_data] )
    #
    #     self._temp_function = model
    #     best_point = differential_evolution( self.loss_function, parameter_bounds)
    #
    #     return best_point.x

    def find_initial_guess_scaling(self, model):

        # print(f"In find_initial_guess_scaling with {model.name=}")

        scaling_args_no_sign = self.find_set_initial_guess_scaling(model)

        # the above args, with sizes based on scaling, could each be positive or negative. Find the best one (of 2^dof)
        best_rchisqr = 1e5
        best_grid_point = scaling_args_no_sign

        scaling_args_sign_list = []
        for idx in range( 2**model.dof ) :
            binary_string_of_index = f"0000000000000000{idx:b}"

            new_gridpoint = scaling_args_no_sign.copy()
            for bit, arg in enumerate(new_gridpoint) :
                new_gridpoint[bit] = arg*(1-2*int(binary_string_of_index[-1-bit]))
            scaling_args_sign_list.append(new_gridpoint)

        for point in scaling_args_sign_list:
            model.set_args( *point )
            temp_rchisqr = self.reduced_chi_squared_of_fit(model)
            if temp_rchisqr < best_rchisqr :
                best_rchisqr = temp_rchisqr
                best_grid_point = point

        model.set_args( *best_grid_point )

        return best_grid_point

    def find_set_initial_guess_scaling(self, composite: CompositeFunction):
        # use knowledge of scaling to guess parameter sizes from the characteristic sizes in the data
        charY = (max([datum.val for datum in self._data]) - min([datum.val for datum in self._data])) / 2
        if composite.func.name == "pow0" :
            # this typically represents a shift, so the average X is more important than the range of x-values
            charX = ( max( [datum.pos for datum in self._data] ) + min( [datum.pos for datum in self._data] ) ) / 2
        else :
            charX = (max([datum.pos for datum in self._data]) - min([datum.pos for datum in self._data])) / 2

        composite.func.arg = 1
        if composite.parent is None :
            # function of this node A*func(x) should scale like y
            composite.func.arg *= charY
        elif composite.parent.func.name == "my_cos" and composite.func.name == "pow1" :
            if len(self._cos_freq_list_dup) > 0 :
                charX = 1 / ( 2*math.pi*self._cos_freq_list_dup.pop(0) )
            else :  # misassigned cosine frequency
                self._sin_freq_list_dup.remove( self._all_freq_list_dup[0] )
                charX = 1 / (2 * math.pi * self._all_freq_list_dup.pop(0))
        elif composite.parent.func.name == "my_sin" and composite.func.name == "pow1" :
            if len(self._sin_freq_list_dup) > 0 :
                charX = 1 / ( 2*math.pi*self._cos_freq_list_dup.pop(0) )
            else :  # misassigned sine frequency
                self._cos_freq_list_dup.remove( self._all_freq_list_dup[0] )
                charX = 1 / (2 * math.pi * self._all_freq_list_dup.pop(0))

        composite.func.arg *= charX ** composite.dimension_arg

        for child in composite.children_list :
            self.find_set_initial_guess_scaling(child)

        return composite.get_args()

    def show_fit(self, model=None):

        x_points = []
        y_points = []
        sigma_x_points = []
        sigma_y_points = []

        for datum in self._data :
            x_points.append( datum.pos )
            y_points.append( datum.val )
            sigma_x_points.append( datum.sigma_pos )
            sigma_y_points.append( datum.sigma_val )

        plot_model = None
        if model is not None:
            plot_model = model.copy()
        else:
            plot_model = self._best_function

        smooth_x_for_fit = np.linspace( x_points[0], x_points[-1], 4*len(x_points))
        fit_vals = [ plot_model.eval_at(xi) for xi in smooth_x_for_fit ]

        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
        plt.errorbar( x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        plt.plot( smooth_x_for_fit, fit_vals, '-', color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        axes = plt.gca()
        if axes.get_xlim()[0] > 0 :
            axes.set_xlim( [0, axes.get_xlim()[1]] )
        if axes.get_ylim()[0] > 0 :
            axes.set_ylim( [0, axes.get_ylim()[1]] )
        axes.set_facecolor( (112/255, 146/255, 190/255) )
        print("Here once")
        plt.show()

    # def save_fit_image(self, filepath, x_label="x", y_label="y", model=None):
    #
    #     x_points = []
    #     y_points = []
    #     sigma_x_points = []
    #     sigma_y_points = []
    #
    #     for datum in self._data :
    #         x_points.append( datum.pos )
    #         y_points.append( datum.val )
    #         sigma_x_points.append( datum.sigma_pos )
    #         sigma_y_points.append( datum.sigma_val )
    #
    #     if model is not None:
    #         plot_model = model.copy()
    #     else:
    #         plot_model = self._best_function
    #
    #     smooth_x_for_fit = np.linspace( x_points[0], x_points[-1], 4*len(x_points))
    #     fit_vals = [ plot_model.eval_at(xi) for xi in smooth_x_for_fit ]
    #
    #     plt.close()
    #     fig = plt.figure()
    #     fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
    #     plt.errorbar( x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
    #     plt.plot( smooth_x_for_fit, fit_vals, '-', color='r' )
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     axes = plt.gca()
    #     if axes.get_xlim()[0] > 0 :
    #         axes.set_xlim( [0, axes.get_xlim()[1]] )
    #     if axes.get_ylim()[0] > 0 :
    #         axes.set_ylim( [0, axes.get_ylim()[1]] )
    #     axes.set_facecolor( (112/255, 146/255, 190/255) )
    #     plt.savefig(filepath)

    # def fit_many_data_sets(self):
    #     pass

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


    # This loss function is used for an initial fit -- we minimize residuals of both the function's position
    # AND the function's derivative
    # The uncertainties and exact fit are unnecessary for an initial guess
    def loss_function(self, par_tuple):
        self._temp_function.set_args(*par_tuple)
        r_sqr = 0
        for datum in self.smoothed_data(n=1) :
            r_sqr += ( self._temp_function.eval_at(datum.pos) - datum.val )**2
        for deriv in self.deriv_on_n_smoothed(n=3) :
            r_sqr += ( self._temp_function.eval_deriv_at(deriv.pos) - deriv.val )**2
        return r_sqr

    def chi_squared_of_fit(self,model):
        chisqr = 0
        for datum in self._data :
            chisqr += ( model.eval_at(datum.pos) - datum.val )**2 / (datum.sigma_val + 1e-5)**2
        return chisqr

    def reduced_chi_squared_of_fit(self,model, use_errors=True):
        k = model.dof
        N = len(self._data)
        return self.chi_squared_of_fit(model) / (N-k) if N > k else 1e5

    def r_squared(self, model):
        mean = sum( [datum.val for datum in self._data] )/len(self._data)
        variance_data = sum( [ (datum.val-mean)**2 for datum in self._data ] )/len(self._data)
        variance_fit = sum( [ (datum.val-model.eval_at(datum.pos))**2 for datum in self._data ] )/len(self._data)
        return 1 - variance_fit/variance_data

    # the AIC is equivalent, for normally distributed residuals, to the least chi squared
    def Akaike_criterion(self, model):
        AIC = self.chi_squared_of_fit(model)
        return AIC

    # correction for small datasets, fixes overfitting
    def Akaike_criterion_corrected(self, model):
        k = model.dof
        N = len(self._data)
        AICc = self.chi_squared_of_fit(model) + 2*k*(k+1)/(N-k-1) if N > k + 1 else 1e5
        return AICc

    # the same as AIC but penalizes additional parameters more heavily for larger datasets
    def Bayes_criterion(self, model):
        k = model.dof
        N = len(self._data)
        AIC = self.Akaike_criterion(model)
        BIC = AIC + k*math.log(N) - 2*k
        return BIC

    # agrees with AIC at small datasets, but punishes less strongly than Bayes at large N
    def HannanQuinn_criterion(self, model):
        k = model.dof
        N = len(self._data)
        AIC = self.Akaike_criterion(model)
        HQIC = AIC + 2*k*math.log( math.log(N) ) - 2*k
        return HQIC


    def smoothed_data(self, n=1):

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
            effective_uncertainty_squared = ratio*sample_uncertainty_squared + (1-ratio)*propagation_uncertainty_squared

            averaged_data.append( Datum1D(pos=key, val=val,
                                          sigma_pos=math.sqrt( propagation_variance_x_dict[key] ),
                                          sigma_val=math.sqrt( effective_uncertainty_squared )
                                         )
                                )
        self._averaged_data = sorted(averaged_data)


