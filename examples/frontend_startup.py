
# internal classes
from autofit.src.frontend import Frontend


def start_frontend():

    gui = Frontend()
    gui.exist()


if __name__ == "__main__" :

    start_frontend()


"""
Packaging instructions for standalone version, no saved settings functionality, long startup 

> pyinstaller --windowed --onefile --hidden-import autofit frontend_startup.py

then change datas in the .spec file to include the data you want 

datas=[('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/icon.ico','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/splash.png','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/plots','plots'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/data','data')],

> pyinstaller frontend_startup.spec

"""

"""
Packaging instructions for directory version, saved settings enabled, quick startup 

> pyinstaller --windowed --hidden-import autofit frontend_startup.py

then change datas in the .spec file to include the data you want 

datas=[('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/icon.ico','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/splash.png','.'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/plots','plots'),
           ('C:/Users/Matt/Documents/GitHub/MIW_AutoFit/autofit/data','data')],

> pyinstaller frontend_startup.spec

"""

