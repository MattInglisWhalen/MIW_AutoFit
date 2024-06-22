"""
MIW's AutoFit provides a powerful backend architecture
with a convenient frontend GUI for fitting and evaluating
many different functional models to a dataset.

This file allows the package to be used as

from autofit import *

which exposes the core classes and methods to the user.
"""

from autofit.src.composite_function import CompositeFunction
from autofit.src.datum1D import Datum1D
from autofit.src.frontend import Frontend
from autofit.src.data_handler import DataHandler
from autofit.src.optimizer import Optimizer
from autofit.src.package import logger, debug, performance, pkg_path
from autofit.src.primitive_function import PrimitiveFunction
