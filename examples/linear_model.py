
# built-in libraries
import math
import random as rng

# external libraries
import numpy as np



if __name__ == "__main__" :

    for x in range(50) :
        print( f"{x - 25},{rng.normalvariate(mu=(10 if x-25 > -7 else -17), sigma=2)}" )


