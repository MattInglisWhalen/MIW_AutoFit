"""Tests MIW's AutoFit GUI class"""

# uilt-in libraries

# external packages

# internal classes
from autofit.src.frontend import Frontend


def test_frontend_startup():
    gui = Frontend()
    gui.shutdown()
