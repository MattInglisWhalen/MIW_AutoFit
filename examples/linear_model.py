
# built-in libraries
import math
import random as rng

# external libraries
import numpy as np



if __name__ == "__main__" :

    for x in range(0,500000,10000) :
        print( f"{x},{rng.normalvariate(mu=0.010*np.sin(np.pi*x/200000.), sigma=0.001)}, 0.002" )


