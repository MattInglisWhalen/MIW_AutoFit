
# built-in libraries
import math
import random as rng

# external libraries
import numpy as np

# internal classes
from autofit.src.optimizer import Optimizer
from autofit.src.datum1D import Datum1D


def list_of_random_positions_in_range(n=50, x_low=0., x_high=10.):
    return [rng.random()*(x_high-x_low)+x_low for _ in range(n)]

def list_of_uniformly_spaced_positions_in_range(n=50, x_low=0., x_high=10.):
    return np.linspace(x_low,x_high,n)

def list_of_random_linear_data(n=50, slope=5., intercept=7., sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=slope*positions[i] + intercept, sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    return data

def list_of_random_cosine_data(n=50, amplitude=7., omega=3., height=5., sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*math.sin(omega*positions[i]) + height, sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    return data

def list_of_random_sin_cosine_data(n=50, amplitude=13, omega1=0.1, omega2=0.5, sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*( math.sin(omega1*positions[i])
                                                 + math.cos(omega2*positions[i]) ), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    return data

def list_of_random_exp_data(n=50, amplitude=13,  decay=3., sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*math.exp(-decay*positions[i]), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    return data

def list_of_random_log_data(n=50, amplitude=13, x0=1., sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*math.log(positions[i]/x0), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    return data

def list_of_random_logistic_data(n=50, amplitude=13, width=5., x0=1., sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude/(1+math.exp(-width*(positions[i]-x0))), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    return data



def test_linear_fit():

    data = list_of_random_linear_data(n=500,slope=7,intercept=5,sigma=5)
    opt = Optimizer(data = data, use_trig=True, use_exp=False, use_powers=False, max_functions=3)
    opt.fit_single_data_set()
    opt.show_fit()

def test_cosine_fit():

    data = list_of_random_cosine_data(amplitude=7,omega=0.5,
                                        height=5.,sigma=1)
    opt = Optimizer(data = data, use_trig=True, use_exp=False, use_powers=False, max_functions=4)
    opt.fit_single_data_set()
    opt.show_fit()

def test_sin_cosine_fit():

    data = list_of_random_sin_cosine_data(amplitude=7,omega1=0.1, omega2=0.5, sigma=2)
    opt = Optimizer(data = data, use_trig=True, use_exp=False, use_powers=False, max_functions=5)
    opt.fit_single_data_set()
    opt.show_fit()

def test_exp_fit():

    data = list_of_random_exp_data(amplitude=17,decay=1/10, sigma=0.5)
    opt = Optimizer(data = data, use_trig=False, use_exp=True, use_powers=False, max_functions=3)
    opt.fit_single_data_set()
    opt.show_fit()

def test_log_fit():

    data = list_of_random_log_data(amplitude=7,x0=3., sigma=2)
    opt = Optimizer(data = data, use_trig=False, use_exp=True, use_powers=False, max_functions=3)
    opt.fit_single_data_set()
    opt.show_fit()

def test_logistic_fit():

    data = list_of_random_logistic_data(amplitude=7,width=3, x0=5, sigma=2)
    opt = Optimizer(data = data, use_trig=False, use_exp=True, use_powers=True, max_functions=5)
    opt.fit_single_data_set()
    opt.show_fit()


if __name__ == "__main__" :

    # test_linear_fit()
    # test_cosine_fit()
    # test_sin_cosine_fit()
    # test_exp_fit()
    # test_log_fit()
    test_logistic_fit()
