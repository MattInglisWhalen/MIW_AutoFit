
# built-in libraries
import math
import random as rng

# external libraries
import numpy as np



if __name__ == "__main__" :

    for _ in range(500) :
        print( rng.normalvariate(mu=10,sigma=3) )

    for _ in range(500) :
        print( rng.normalvariate(mu=20,sigma=3) )