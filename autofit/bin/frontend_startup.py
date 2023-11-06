"""For running MIW's AutoFit as an interactive GUI"""

# default libraries
# ---

# external libraries
# ---

# internal classes
from autofit.src.frontend import Frontend


########################################################################################################################


def start_frontend():
    gui = Frontend()
    gui.exist()


if __name__ == "__main__" :
    start_frontend()


"""
Packaging instructions

> pyinstaller --windowed --hidden-import autofit MIW_autofit.py

then change datas in the .spec file to include the data you want 

datas=[('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/icon.ico','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/splash.png','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/plots','plots'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/data','data')],

> pyinstaller MIW_autofit.spec

"""
