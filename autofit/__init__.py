"""
MIW's AutoFit provides a powerful backend architecture
with a convenient frontend GUI for fitting and evaluating
many different functional models to a dataset.

This file allows the package to be used as

from autofit import *

which exposes the core classes and methods to the user.
"""

from .src.composite_function import CompositeFunction
from .src.datum1D import Datum1D
from .src.frontend import Frontend
from .src.data_handler import DataHandler
from .src.optimizer import Optimizer
from .src.package import logger, debug, performance
from .src.primitive_function import PrimitiveFunction
