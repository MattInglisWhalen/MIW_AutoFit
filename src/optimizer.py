
# default libraries
import math
from dataclasses import field
import re as regex
from collections import defaultdict

# external libraries
import numpy
from scipy.optimize import curve_fit, differential_evolution
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

    def __init__(self, use_trig = False, use_powers = False, use_exp = False, data=None, max_functions=5):

        # datasets, which are lists of Datum1D instances
        self._data = []             # the raw datapoints of (x,y), possibly with uncertainties
                                    # May also represent a binned point if the input file is a list of x_values only
        self._averaged_data = []    # the model optimizer requires approximate slope knowledge,
                                    # so multiple measurements at each x-value are collapsed into a single value

        # fit results
        self._best_function = None          # a CompositeFunction
        self._best_args = None              # the arguments (parameters) of best_function CompositeFunction
        self._best_args_uncertainty = None  # the uncorrelated uncertainties of the arguments

        self._top_5_models = []
        self._top_5_args = []
        self._top_5_uncertainties = []

        # function construction parameters
        self._max_functions = max_functions
        self._primitive_function_list = []

        # useful auxiliary varibles
        self._temp_function = None      # a CompositeFunction

        if data is not None :
            self._data = sorted(data)  # list of Datum1D. Sort makes the data monotone increasing in x-value
            self.average_data()

        self.load_default_functions()
        if use_trig :
            self.load_trig_functions()
        if use_powers :
            self.load_power_functions()
        if use_exp :
            self.load_exp_functions()

        self._composite_function_list = []


    def __repr__(self):
        pass

    @property
    def best_model(self):
        return self._best_function
    @property
    def parameters(self):
        return self._best_args
    @property
    def uncertainties(self):
        return self._best_args_uncertainty

    def build_composite_function_list1(self):

        # start with simple primitives
        for iprim in self._primitive_function_list :
            new_comp = CompositeFunction( func=iprim )
            self._composite_function_list.append( new_comp )

        for icomp in self._composite_function_list[:]:
            # composition
            for iprim in self._primitive_function_list:
                new_comp = CompositeFunction(func=iprim, children_list=[icomp])
                self._composite_function_list.append(new_comp)

        for depth in range(self._max_functions) :
            for icomp in self._composite_function_list[:] :
                # composition
                for iprim in self._primitive_function_list :
                    new_comp = CompositeFunction(func=iprim, children_list=[icomp])
                    self._composite_function_list.append(new_comp)
                # addition
                for iprim in self._primitive_function_list[:] :
                    new_comp = icomp.copy()
                    new_comp.add_child(iprim)
                    self._composite_function_list.append(new_comp)

        print(f"Pre-trimmed list: (len={len(self._composite_function_list)})")
        for icomp in self._composite_function_list :
            # print(icomp)
            pass
        print("|----------------\n")


        self.trim_composite_function_list()
        print(f"After trimming list: (len={len(self._composite_function_list)})")
        for icomp in self._composite_function_list :
            print(icomp)
        print("|----------------\n")

    def build_composite_function_list(self):

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

        print(f"Pre-trimmed list: (len={len(self._composite_function_list)})")
        for icomp in self._composite_function_list:
            # print(icomp)
            pass
        print("|----------------\n")

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

        # need to know which names we're working with
        prim_names_list = []
        for prim in self._primitive_function_list :
            prim_names_list.append(prim.name)

        num_comps = len(self._composite_function_list[:])

        # use regex to trim based on rules applied to composite names
        for idx, icomp in enumerate(self._composite_function_list[:]) :

            if idx % 50 == 0 :
                print(f"{idx}/{num_comps}")

            remove_flag = 0
            name = icomp.name

            # composition of a constant function is wrong
            if regex.search( f"pow0\(" , name):
                remove_flag = 2
            if regex.search( f"\(pow0\)" , name):
                remove_flag = 3

            # composition of powers with no sum is wrong
            if regex.search( f"pow[0-9]\(pow[a-z0-9_]*\)" , name):
                remove_flag = 5

            # repeated reciprocal is wrong
            if regex.search( f"pow_neg1\(pow_neg1\)" , name):
                remove_flag = 7

            # trivial reciprocal is wrong
            if regex.search( f"pow_neg1\(pow1\)" , name):
                remove_flag = 11

            # composition of pow1 with no sum is wrong
            for prim_name in prim_names_list :
                if regex.search( f"pow1\({prim_name}\)" , name):
                    remove_flag = 13

            # deeper composition of pow1 with no sum is wrong
            for prim_name1 in prim_names_list :
                for prim_name2 in prim_names_list:
                    if regex.search( f"pow1\({prim_name1}\({prim_name2}\)\)" , name):
                        remove_flag = 17

            # sum of pow1 with composition following is wrong
            if regex.search( f"\+pow1\(" , name):
                remove_flag = 19

            # sum of the same function is wrong
            for prim_name in prim_names_list :
                if regex.search( f"{prim_name}[a-z+]*{prim_name}" , name):
                    remove_flag = 23

            # trig inside trig is too complex, and very (very) rarely occurs in applications
            if icomp.has_double_trigness() :
                remove_flag = 29
            if icomp.has_double_expness() :
                remove_flag = 29
            if icomp.has_double_logness() :
                remove_flag = 29

            if name[0:4] == "pow1" and icomp.num_children() == 1:
                remove_flag = 31



            # +log(pow1) is a duplicate since ...+Alog(Bx) = ...+ AlogB + Alog(x) = ... + my_pow0 + my_log
            if regex.search( f"\+my_log\(pow1\)" , name) or regex.search( f"\(my_log\(pow1\)" , name):
                remove_flag = 37

            # sins and cosines and logs and exp never have angular frequency or decay parameters exactly 1
            # I'd like to do the same with log but that conflicts with rule 37
            if regex.search( f"sin\)" , name) or regex.search( f"cos\)" , name) or regex.search( f"exp\)" , name)   :
                remove_flag = 41
            if regex.search( f"sin\+" , name) or regex.search( f"cos\+" , name) or regex.search( f"exp\+" , name) :
                remove_flag = 43

            # trivial composition should just be straight sums
            if regex.search( f"\(pow1\(", name ) or regex.search( f"\+pow1\(", name ):
                remove_flag = 47

            # more more exp algebra: Alog( Bexp(Cx) ) = AlogB + ACx = pow0+pow1
            if regex.search( f"my_log\(my_exp\(pow1\)\)", name ):
                remove_flag = 53

            # reciprocal exponent is already an exponent pow_neg1(exp) == my_exp(pow1)
            if regex.search( f"pow_neg1\(my_exp\)", name ):
                remove_flag = 61
            # and again but deeper
            for prim_name in prim_names_list :
                if regex.search(f"pow_neg1\(my_exp\({prim_name}\)\)", name):
                    remove_flag = 61

            # log of "higher" powers \is the same as log + pow0
            if regex.search( f"my_log\(pow[a-z0-9_]*\)", name ):
                remove_flag = 67

            # exp(pow0+...) is just exp(...)
            if regex.search( f"my_exp\(pow0", name ):
                remove_flag = 71
            # and again
            for prim_name in prim_names_list :
                if regex.search(f"my_exp\({prim_name}+pow0", name):
                    remove_flag = 71

            # log(1/f+g) is the same as log(f+g)
            for prim_name1 in prim_names_list :
                if regex.search(f"my_log\(pow_neg1\({prim_name1}\)\)", name):
                    remove_flag = 73
                for prim_name2 in prim_names_list:
                    if regex.search( f"my_log\(pow_neg1\({prim_name1}\+{prim_name2}\)\)" , name):
                        remove_flag = 73

            # repeated pow_neg1 is dumb -- should put all these loops together
            for prim_name1 in prim_names_list :
                if regex.search(f"pow_neg1\(pow_neg1\({prim_name1}\)\)", name):
                    remove_flag = 79
                for prim_name2 in prim_names_list:
                    if regex.search( f"pow_neg1\(pow_neg1\({prim_name1}\+{prim_name2}\)\)" , name):
                        remove_flag = 79
                    for prim_name3 in prim_names_list:
                        if regex.search(f"pow_neg1\(pow_neg1\({prim_name1}\+{prim_name2}\+{prim_name3}\)\)", name):
                            remove_flag = 79


            if name == "pow1(my_cos(pow1)+my_sin(pow1))" :
                if remove_flag :
                    print(f"Why did we remove {name=} at {remove_flag=}")
                    raise SystemExit

            if name == "pow_neg1" :
                if remove_flag :
                    print(f"Why did we remove {name=} with {remove_flag=}")
                    raise SystemExit

            if remove_flag :
                self._composite_function_list.remove(icomp)

        # remove duplicates using a dict{}
        self._composite_function_list = list({ icomp.name : icomp for icomp in self._composite_function_list[:] }.values())

    def add_primitive_to_list(self):
        # For adding one's own Primitive Functions to the built-in list
        pass

    def find_function(self):
        pass

    def parameters_and_uncertainties_from_fitting(self, model, initial_guess = None):

        print(f"{model=} {model.dof=}")

        x_points = []
        y_points = []
        sigma_points = []

        use_errors = True

        for datum in self._data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            if datum.sigma_val < 1e-5:
                use_errors = False
                sigma_points.append(1.)
            else:
                sigma_points.append(datum.sigma_val)

        # Find an initial guess for the parameters based off scipy's genetic algorithm
        # The loss function there also tries to fit the data's smoothed derivatives
        if initial_guess is None :
            initial_guess = self.find_initial_guess_genetic(model)
            model.set_args(*initial_guess)

        # Next, find a better guess by relaxing the error bars on the data
        # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
        # and so relaxed error bars help point towards global minima
        better_guess, _ = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points, p0=initial_guess, maxfev=5000)

        # Finally, use the better guess to find the true minimum with the true error bars
        np_pars, np_cov = curve_fit(model.scipy_func, xdata=x_points, ydata=y_points,
                                    sigma=sigma_points, absolute_sigma=use_errors,
                                    p0=better_guess, maxfev=5000)

        pars = np_pars.tolist()
        uncertainties = np.sqrt(np.diagonal(np_cov)).tolist()
        model.set_args(*pars)  # ignore iterable

        self._best_function = model.copy()
        self._best_args = pars  # have to do this because the model's stored parameters change for some reason
        self._best_args_uncertainty = uncertainties

        return pars, uncertainties

    def find_best_model_for_dataset(self):

        self.build_composite_function_list()

        best_LXsqr = 1e5

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
            else:
                sigma_points.append( datum.sigma_val )

        num_models = len(self._composite_function_list)
        for idx, model in enumerate(self._composite_function_list) :

            np_pars, np_cov = None, None    # to be found

            # Find an initial guess for the parameters based off scipy's genetic algorithm
            # The loss function there also tries to fit the data's smoothed derivatives
            initial_guess = self.find_initial_guess_genetic(model)
            model.set_args(*initial_guess)
            print(f"{idx+1}/{num_models} Genetic guess:")
            model.print_tree()

            # Next, find a better guess by relaxing the error bars on the data
            # Unintuitively, this helps. Tight error bars flatten the gradients away from the global minimum,
            # and so relaxed error bars help point towards global minima
            try:
                better_guess, _ = curve_fit( model.scipy_func, xdata=x_points, ydata=y_points,
                                             p0=initial_guess, maxfev=5000)
            except RuntimeError :
                print("Couldn't find optimal parameters. Continuing on...")
                continue

            # Finally, use the better guess to find the true minimum with the true error bars
            try:
                np_pars, np_cov = curve_fit( model.scipy_func, xdata=x_points, ydata=y_points,
                                             sigma=sigma_points, absolute_sigma=use_errors,
                                             p0=better_guess, maxfev=5000)
            except RuntimeError :
                print("Couldn't find optimal parameters. Continuing on...")
                continue

            pars = np_pars.tolist()
            uncertainties = np.sqrt(np.diagonal(np_cov)).tolist()
            model.set_args( *pars )     # ignore iterable

            # goodness of fit: reduced chi_R^2 = chi^2 / (N-k) should be close to 1
            # chi_R^2 = 0.001 is overfit, chi_R^2 = 1000 is a bad fit
            logsqr_of_models_reduced_chisqr = ( math.log( self.reduced_chi_squared_of_fit(model) ) )**2

            # alternatively, there are other information criteria which punish additional parameters differently
            AICc = self.Akaike_criterion_corrected(model)
            BIC = self.Bayes_criterion(model)
            HQIC = self.HannanQuinn_criterion(model)

            if model.name == "pow1(my_cos(pow1)+my_sin(pow1))" :
                print("\n <-----------> what's wrong with pow1(my_cos(pow1)+my_sin(pow1))???")
                print(f"{pars=} {uncertainties=} red_chisqr={self.reduced_chi_squared_of_fit(model)}")
                print(f"Other criterions: {AICc=}, {BIC=}, {HQIC=}")
                model.print_tree()
                self.show_fit(model=model)

            if logsqr_of_models_reduced_chisqr < best_LXsqr \
                    and not numpy.isinf( uncertainties[0] ) and uncertainties[0]**2 < 1e5 :
                best_LXsqr = logsqr_of_models_reduced_chisqr
                self._best_function = model.copy()
                self._best_args = pars  # have to do this because the model's stored parameters change for some reason
                self._best_args_uncertainty = uncertainties
                # self._best_function.track_changes=True
                print(f"New best with red_chisqr={self.reduced_chi_squared_of_fit(model)}: "
                      f"{model=} with {pars=} +- {uncertainties=}")
                print(f"Other criterions: {AICc=}, {BIC=}, {HQIC=}")
                self._best_function.print_tree()


        print(f"\nBest model is {self._best_function} "
              f"\n with args {self._best_args} += {self._best_args_uncertainty} "
              f"\n and reduced chi-sqr {math.exp(math.sqrt(best_LXsqr))}")
        self._best_function.set_args( *self._best_args )  # ignore iterable
        self._best_function.print_tree()


    def find_initial_guess(self, model: CompositeFunction, width = 10., num_each=10):

        best_rchisqr = 1e5
        best_grid_point = model.get_args()

        total_grid = np.mgrid[tuple(slice(-width / 2, width / 2, complex(0, num_each)) for _ in range(model.dof))]

        args = []
        for idim in range( model.dof ):
            args.append(total_grid[idim].flatten())
        for ipoint in zip(*args):
            model.set_args( *ipoint )
            temp_rchisqr = self.reduced_chi_squared_of_fit(model)
            if temp_rchisqr < best_rchisqr :
                best_rchisqr = temp_rchisqr
                best_grid_point = list(ipoint)

        model.set_args( *best_grid_point)

        print("\n Grid guess:")
        model.print_tree()
        return best_grid_point

    def find_initial_guess_genetic(self, model: CompositeFunction):

        max_Y_data = max( [datum.val for datum in self._data] )
        max_X_data = max( [datum.pos for datum in self._data] )
        max_data = max( max_X_data, max_Y_data )

        parameter_bounds = []
        for _ in range(model.dof):
            parameter_bounds.append( [-max_data, max_data] )

        self._temp_function = model
        best_point = differential_evolution( self.loss_function, parameter_bounds)

        return best_point.x


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

        fig = plt.figure()
        fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
        plt.errorbar( x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        plt.plot( smooth_x_for_fit, fit_vals, '-', color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        axes = plt.gca()
        axes.set_facecolor( (112/255, 146/255, 190/255) )
        plt.show()

    def save_fit_image(self, filepath, x_label="x", y_label="y", model=None):

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
        plt.plot( smooth_x_for_fit, fit_vals, '-', color='r' )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        axes = plt.gca()
        axes.set_facecolor( (112/255, 146/255, 190/255) )
        plt.savefig(filepath)


    def fit_many_data_sets(self):
        pass

    def load_default_functions(self):
        self._primitive_function_list.extend( [ PrimitiveFunction.built_in("pow0"),
                                                PrimitiveFunction.built_in("pow1") ] )

    def load_trig_functions(self):
        self._primitive_function_list.extend( [ PrimitiveFunction.built_in("sin"),
                                                PrimitiveFunction.built_in("cos") ] )

    def load_power_functions(self):
        self._primitive_function_list.extend( [ PrimitiveFunction.built_in("pow_neg1") ] )

    def load_exp_functions(self):
        self._primitive_function_list.extend( [ PrimitiveFunction.built_in("exp"),
                                                PrimitiveFunction.built_in("log") ] )
    def load_log_functon(self, rebuild_flag = False):
        self._primitive_function_list.extend([PrimitiveFunction.built_in("log")])
        if rebuild_flag:
            self.build_composite_function_list()
    def unload_log_functon(self):
        self._primitive_function_list.remove([PrimitiveFunction.built_in("log")])
        self.build_composite_function_list()

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

    def reduced_chi_squared_of_fit(self,model):
        k = model.dof
        N = len(self._data)
        return self.chi_squared_of_fit(model) / max(1e-5,N-k)

    # the AIC is equivalent, for normally distributed residuals, to the least chi squared
    def Akaike_criterion(self, model):
        AIC = self.chi_squared_of_fit(model)
        return AIC

    # correction for small datasets, fixes overfitting
    def Akaike_criterion_corrected(self, model):
        k = model.dof
        N = len(self._data)
        AICc = self.chi_squared_of_fit(model) + 2*k*(k+1)/(N-k-1)
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



            # if uncertainties accompany data, use the chi-squared measure
    #  chi^2 := sum_i  ( y_i - f(x_i) )^2 / ( sigma_yi^2 + sigma_xi^2 f'(x_i)^2 )
    # uncertainty on the central value is given by going to chi^2_min -> chi^2_min + 1

    # if no uncertainties are provided, use the R^2 measure
    #   R^2 := sum_i ( y_i - f(x_i) )^2
    # i.e. chi^2 assuming sigma_y = 1 and sigma_x = 0
    # the uncertainty is then ... need to look at least squares model to see how thy derive uncertainty

