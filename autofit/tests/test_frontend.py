"""Tests MIW's AutoFit GUI class"""

# uilt-in libraries

# external packages

# internal classes
from autofit.src.frontend import Frontend
from autofit.tests.conftest import tkinter_available


@tkinter_available
def test_frontend_startup():
    gui = Frontend()
    gui.shutdown()
