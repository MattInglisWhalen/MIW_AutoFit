
# built-in libraries
import math
import random as rng

# external libraries
import numpy as np



if __name__ == "__main__" :

    for x in np.arange(1,50,1.3) :
        x_centre = rng.normalvariate( mu=x, sigma = 0.3)
        print(f"{x_centre:.3F},"
              f"{0.3:.3F},"
              f"{rng.normalvariate(mu=5 * np.exp(-np.log(x/7)**2 / 3), sigma=4/x):.3F},"
              f"{4/x_centre:.3F}")

