
import autofit.examples.frontend_startup as fe_start

from autofit.src.frontend import Validator

def start_autofit():

    validator = Validator()
    error_msg = validator.invalid_config()
    if error_msg != "" :
        Validator.invalid_popup(error_msg)
        raise SystemExit

    fe_start.start_frontend()


if __name__ == "__main__":

    start_autofit()
