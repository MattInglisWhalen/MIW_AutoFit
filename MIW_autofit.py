
import autofit.examples.frontend_startup as fe_start

from autofit.src.validator import Validator

def start_autofit():

    validator = Validator()
    if validator.invalid_config() :
        Validator.invalid_popup()
        raise SystemExit

    fe_start.start_frontend()


if __name__ == "__main__":

    start_autofit()
