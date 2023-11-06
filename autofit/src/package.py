# default libraries
import os
import sys

# external libraries

# internal classes

PACKAGE_PATH : str = ""
DEV = 1  # 1 = print, 0 = log file, -1 = /dev/null

def pkg_path():
    if PACKAGE_PATH == "" :
        _set_package_path()
    return PACKAGE_PATH

def logger(*args) -> None :

    if DEV == 1 :
        print(*args)
    elif DEV == 0 :
        statement = ""
        for idx, arg in enumerate(args) :
            statement += f" {arg}" if idx > 0 else f"{arg}"
        _to_append(statement)
    else :
        pass

def debug(*args) -> None :
    if DEV == 1 :
        print(*args)
    else :
        pass

def _to_append(logstr: str) -> None:
    if PACKAGE_PATH == "" :
        _set_package_path()

    log_filepath = f"{PACKAGE_PATH}/autofit.log"
    with open(file=log_filepath, mode='a+', encoding='utf-8') as log_file :
        log_file.write(f"{logstr}\n")

def _to_clean(logstr: str) -> None:
    if PACKAGE_PATH == "" :
        _set_package_path()
    log_filepath = f"{PACKAGE_PATH}/autofit.log"
    with open(file=log_filepath, mode='w', encoding='utf-8') as log_file :
        log_file.write(f"{logstr}\n")

def _set_package_path() -> None:

    global PACKAGE_PATH
    filepath = ""
    try:
        loc = sys._MEIPASS  # for pyinstaller with standalone exe/app
    except AttributeError:
        filepath = os.path.abspath(__file__)
        loc = os.path.dirname(filepath)

    fallback = loc
    failsafe = 0
    # keep stepping back from the current directory until we are in the directory /autofit
    while loc[-7:] != "autofit":
        failsafe += 1
        loc = os.path.dirname(loc)
        if loc == os.path.dirname(loc):
            print(f"Frontend.get_package_path(): python script >{__file__}<\n"
                  f"nor >{filepath}< is in the AutoFit package's directory.")
            loc = fallback
            break
        if failsafe > 50 :
            print("Frontend.get_package_path(): Failsafe reached in _set_package_path.py")
            loc = fallback
            break

    if sys.platform == "darwin" :
        if os.path.exists(f"{loc}/MIWs_AutoFit.app") :
            loc = loc + "/MIWs_AutoFit.app/Contents/MacOS"
    else :
        if os.path.exists(f"{loc}/backend") :
            loc = loc + "/backend"

    PACKAGE_PATH = loc
    print(PACKAGE_PATH)
    _to_clean(PACKAGE_PATH)
