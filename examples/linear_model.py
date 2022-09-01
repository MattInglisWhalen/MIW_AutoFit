
# built-in libraries
import math
import random as rng

# external libraries
import numpy as np

# internal classes
from autofit.src.optimizer import Optimizer
from autofit.src.datum1D import Datum1D
from autofit.src.primitive_function import PrimitiveFunction


def list_of_random_positions_in_range(n=50, x_low=0., x_high=10.):
    return [rng.random()*(x_high-x_low)+x_low for _ in range(n)]

def list_of_uniformly_spaced_positions_in_range(n=50, x_low=0., x_high=10.):
    return np.linspace(x_low,x_high,n)

def list_of_random_normal_data(n=50, mean=5., width=2.):
    positions = [ rng.normalvariate( mu=mean, sigma=width) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=0) for i in range(n) ]
    return data

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
    for datum in data :
        print(f"{datum.pos}, {datum.val}, {datum.sigma_val}")
    return data

def list_of_random_expsin_data(n=50, amplitude=13, compression = 0.5, omega1=0.1, sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*math.exp( compression*math.sin(omega1*positions[i]) ), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    for datum in data :
        print(f"{datum.pos}, {datum.val}, {datum.sigma_val}")
    return data

def list_of_random_sinexp_data(n=500, amplitude=13, compression = 0.5, omega1=0.1, sigma=1., x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*math.sin( compression*math.exp(omega1*positions[i]) ), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    for datum in data :
        print(f"{datum.pos}, {datum.val}, {datum.sigma_val}")
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
    values = [ rng.normalvariate( mu=amplitude/(1+math.exp(-(positions[i]-x0)/width)), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma, sigma_pos=0.05) for i in range(n) ]
    return data

def test_histogram_fit():

    data = list_of_random_normal_data(n=50,mean=5,width=2)
    prt_str = ""
    for datum in data :
        prt_str += f"{datum.pos:.2f},"
    print(prt_str)
    # opt = Optimizer(data = data, use_trig=True, use_exp=False, use_powers=False, max_functions=3)
    # opt.fit_single_data_set()
    # opt.show_fit()

def test_linear_fit():

    data1 = list_of_random_linear_data(n=5,slope=1,intercept=2,sigma=0.8,x_low=1,x_high=5)
    data2 = list_of_random_linear_data(n=5,slope=1,intercept=2,sigma=0.8,x_low=1,x_high=5)
    data3 = list_of_random_linear_data(n=5,slope=1,intercept=2,sigma=0.8,x_low=1,x_high=5)
    data4 = list_of_random_linear_data(n=5,slope=1,intercept=2,sigma=0.8,x_low=1,x_high=5)
    data5 = list_of_random_linear_data(n=5,slope=1,intercept=2,sigma=0.8,x_low=1,x_high=5)

    for datum in data1 :
        print(f"{datum.pos}, {0.05}, {datum.val:.2f}, {datum.sigma_val:.2f}")
    for datum in data2 :
        print(f"{datum.pos}, {0.05}, {datum.val:.2f}, {datum.sigma_val:.2f}")
    for datum in data3 :
        print(f"{datum.pos}, {0.05}, {datum.val:.2f}, {datum.sigma_val:.2f}")
    for datum in data4 :
        print(f"{datum.pos}, {0.05}, {datum.val:.2f}, {datum.sigma_val:.2f}")
    for datum in data5 :
        print(f"{datum.pos}, {0.05}, {datum.val:.2f}, {datum.sigma_val:.2f}")
    opt = Optimizer(data = data, use_trig=False, use_exp=False, use_powers=False, max_functions=3)
    opt.find_best_model_for_dataset()
    opt.show_fit()

def test_cosine_fit():

    data = list_of_random_cosine_data(amplitude=7,omega=0.5,
                                        height=5.,sigma=1)
    opt = Optimizer(data = data, use_trig=True, use_exp=False, use_powers=False, max_functions=4)
    opt.find_best_model_for_dataset()
    opt.show_fit()

def test_sin_cosine_fit():

    data = list_of_random_sin_cosine_data(amplitude=7,omega1=0.1, omega2=0.5, sigma=2)
    opt = Optimizer(data = data, use_trig=True, use_exp=False, use_powers=False, max_functions=5)
    opt.find_best_model_for_dataset()
    opt.show_fit()

def test_exp_fit():

    data = list_of_random_exp_data(amplitude=17,decay=1/10, sigma=0.5)
    opt = Optimizer(data = data, use_trig=False, use_exp=True, use_powers=False, max_functions=3)
    opt.find_best_model_for_dataset()
    opt.show_fit()

def test_log_fit():

    data = list_of_random_log_data(amplitude=7,x0=3., sigma=2)
    opt = Optimizer(data = data, use_trig=False, use_exp=True, use_powers=False, max_functions=3)
    opt.find_best_model_for_dataset()
    opt.show_fit()

def test_logistic_fit():

    data = list_of_random_logistic_data(amplitude=7,width=5, x0=20, sigma=1)
    for datum in data :
        print(f"{datum.pos}, {datum.val:.2f}, {datum.sigma_val:.2f}")
    opt = Optimizer(data = data, use_trig=False, use_exp=True, use_powers=True, max_functions=3)
    opt.find_best_model_for_dataset()
    opt.show_fit()

def create_power_law_data(n=50, amplitude=0.5, power=1.5, sigma=1, x_low=1, x_high=50):
    positions = list_of_uniformly_spaced_positions_in_range(n=n, x_low=x_low, x_high=x_high)
    values = [ rng.normalvariate( mu=amplitude*np.power(positions[i],power), sigma=sigma) for i in range(n) ]
    data = [ Datum1D(pos=positions[i], val=values[i], sigma_val=sigma) for i in range(n) ]
    for datum in data:
        print(f"{datum.pos:.2}, {datum.val:.2}, {datum.sigma_val}")


if __name__ == "__main__" :
    list_of_random_sinexp_data()
    # list_of_random_expsin_data()
    # list_of_random_sin_cosine_data()
    # create_power_law_data()
    # test_histogram_fit()
    # test_linear_fit()
    # test_cosine_fit()
    # test_sin_cosine_fit()
    # test_exp_fit()
    # test_log_fit()
    # test_logistic_fit()
