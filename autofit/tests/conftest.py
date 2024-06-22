"""Configuration for pytest"""

# built-ins

# external packages
import pytest
import numpy as np
import tkinter as tk
import _tkinter as tk_hidden

# internal classes


def tkinter_available(func):
    try:
        tk.Tk().destroy()

        def wrapper_with_tk():
            func()

        return wrapper_with_tk

    except tk_hidden.TclError:

        def wrapper_without_tk():
            pass

        return wrapper_without_tk


def assert_almost_equal(a, b, tol=1e-5, recurse=2):
    """
    Checks for the near-equality of two numerical variables.
    If they are subscriptable, the entries are checked elementwise.
    """
    if recurse > 1 and len(np.shape(a)) > 1:
        assert a.shape == b.shape
        assert_almost_equal(a.flatten(), b.flatten(), recurse=1)
        return

    if recurse > 0 and len(np.shape(a)) == 1:
        assert len(a) == len(b)
        for A, B in zip(a, b):
            assert_almost_equal(A, B, recurse=0)
        return

    if np.isinf(a) and np.isinf(b):
        return

    if np.isnan(a) and np.isnan(b):
        return

    mean = tol + abs((a + b) / 2)
    diff = abs(a - b)

    assert diff / mean < tol
