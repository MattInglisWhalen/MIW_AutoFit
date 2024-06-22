"""Tests the benchmarking utility"""

# built-ins

# external libraries

# internal classes
from autofit.tests.benchmarking.evaluate_accuracy import find_accuracy_of_optimizer


def test_depth_1():

    find_accuracy_of_optimizer(max_depth=1)
