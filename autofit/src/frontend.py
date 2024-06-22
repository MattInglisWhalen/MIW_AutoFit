"""Implements the Tkinter logic for GUI execution"""

# default libraries
import sys
import os.path as ospath
import re as regex
from functools import partial
from typing import Union

# external packages
import tkinter as tk
import tkinter.filedialog as fd

import numpy as np
from pandas import ExcelFile

import scipy.stats
import scipy.special
from PIL import Image
# if sys.platform == "linux":
#     import matplotlib
#
#     # use this with  hiddenimports=['PIL', 'PIL._imagingtk', 'PIL._tkinter_finder']
#     matplotlib.use("TkAgg")
#     # if that doesn't work, in optimizer.show_fit() change
#     #       plt.show(block=pause_on_image)
#     # to
#     # if sys.platform == "linux" :
#     #       import subprocess
#     #       plt.savefig(f"{pkg_path()}/plots/residuals", facecolor=...)
#     #       subprocess.call('xdg-open',f"{pkg_path()}/plots/residuals")
#     # else :
#     #       plt.show(block=pause_on_image)
#     #
#     # and also remove the inspect button
import matplotlib.pyplot as plt

# internal classes
from autofit.src.composite_function import CompositeFunction
from autofit.src.primitive_function import PrimitiveFunction
from autofit.src.data_handler import DataHandler
from autofit.src.optimizer import Optimizer
from autofit.src.package import pkg_path, logger
import autofit.src.plot_formatter as pf


# TODO:
# also Ubuntu, need matplotlib.use('TkAgg').
#         -> TkAgg doesnt work right now, but can try
#         hiddenimports=['PIL', 'PIL._imagingtk', 'PIL._tkinter_finder']
#         This should avoid "UserWarning: matplotlib is currently using agg, which is a non-gui
#         backend, cannot show the figure"
# ~ Possible todos for pro version ~
# Copy to clipboard console
# Reinstate logging capability
# Finer detail sig figs of parameters
# Covariance matrix
# Residuals under plot


class Frontend:
    """The singleton class for MIW's AutFit GUI"""

    _meipass_flag = True

    def __init__(self):
        """Begins the GUI loop"""

        self.set_meipass()

        # UX
        self._new_user_stage: int = (
            1  # uses prime factors to notate which actions the user has taken
        )

        # UI
        self._gui: tk.Tk = tk.Tk()
        self._os_width: int = self._gui.winfo_screenwidth()
        self._os_height: int = self._gui.winfo_screenheight()

        # file handling
        self._filepaths: list[str] = []
        self._data_handlers: list[DataHandler] = []
        self._changed_data_flag: bool = True

        # backend connections
        self._optimizer: Union[None, Optimizer] = None
        self._changed_optimizer_opts_flag: bool = True

        # panels
        self._left_panel_frame: Union[None, tk.Frame] = None
        self._middle_panel_frame: Union[None, tk.Frame] = None
        self._right_panel_frame: Union[None, tk.Frame] = None

        # left panel ------------------------------------------------------------------------------>
        self._fit_data_button: Union[None, tk.Button] = None

        # Excel input
        self._popup_window: Union[None, tk.Toplevel] = None
        self._excel_x_range: str = ""
        self._excel_y_range: str = ""
        self._excel_sigmax_range: str = ""
        self._excel_sigmay_range: str = ""
        self._all_sheets_in_file: tk.BooleanVar = tk.BooleanVar(value=False)

        # right panel ----------------------------------------------------------------------------->
        self._max_message_length = 20
        self._num_messages_ever: int = 0
        self._num_messages: int = 0

        self._colors_console_menu: Union[None, tk.Menu] = None
        self._console_color: str = "black"
        self._printout_color: tuple[int, int, int] = (0, 200, 0)

        self._progress_label = None

        # middle panel ---------------------------------------------------------------------------->
        self._data_perusal_frame: Union[None, tk.Frame] = None
        self._fit_options_frame: Union[None, tk.Frame] = None
        self._plot_options_frame: Union[None, tk.Frame] = None
        self._polynomial_frame: Union[None, tk.Frame] = None
        self._procedural_frame: Union[None, tk.Frame] = None
        self._gaussian_frame: Union[None, tk.Frame] = None
        self._manual_frame: Union[None, tk.Frame] = None

        # image frame
        self._curr_image_num: int = -1
        self._image_path: str = ""
        self._image: Union[None, tk.PhotoImage] = None
        self._image_frame: Union[None, tk.Label] = None

        self._showing_fit_image: bool = False  # conjugate to showing data-only image
        self._showing_fit_all_image: bool = False
        self._bg_color: tuple[float, float, float] = (112 / 255, 146 / 255, 190 / 255)
        self._fit_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
        self._dataaxes_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._color_name_tkstr: tk.StringVar = tk.StringVar(value="Colour")
        self._colors_image_menu: Union[None, tk.Menu] = None

        # data perusal frame
        self._residuals_button: Union[None, tk.Button] = None
        self._error_bands_button: Union[None, tk.Button] = None
        self._show_error_bands = 0

        # fit options frame
        self._model_name_tkstr: tk.StringVar = tk.StringVar(value="")
        self._which5_name_tkstr: Union[None, tk.StringVar] = None
        self._which_tr_id = None

        self._pause_button: Union[None, tk.Button] = None
        self._refit_button: Union[None, tk.Button] = None
        self._refit_on_click = True

        # plot options frame
        self._logx_button: Union[None, tk.Button] = None
        self._logy_button: Union[None, tk.Button] = None
        self._normalize_button: Union[None, tk.Button] = None

        # polynomial frame
        self._polynomial_degree_tkint: tk.IntVar = tk.IntVar(value=2)
        self._polynomial_degree_label: Union[None, tk.Label] = None

        # gaussian frame
        self._gaussian_modal_tkint: tk.IntVar = tk.IntVar(value=1)
        self._gaussian_modal_label: Union[None, tk.Label] = None

        # procedural frame
        self._checkbox_names_list = [
            "cos(x)",
            "sin(x)",
            "exp(x)",
            "log(x)",
            "1/x",
            "custom",
        ]
        self._use_func_dict_name_tkbool = {}  # for checkboxes
        for name in self._checkbox_names_list:
            self._use_func_dict_name_tkbool[name] = tk.BooleanVar(value=False)
        self._max_functions_tkint = tk.IntVar(value=3)
        self._depth_label: Union[None, tk.Label] = None
        self._custom_checkbox: Union[None, tk.Checkbutton] = None
        self._custom_binding = None
        self._custom_remove_menu: Union[None, tk.Menu] = None

        # brute-force frame
        self._brute_forcing_tkbool = tk.BooleanVar(value=False)

        # manual frame
        self._manual_name_tkstr: tk.StringVar = tk.StringVar(value="")
        self._manual_form_tkstr: tk.StringVar = tk.StringVar(value="")
        self._manual_model: Union[None, CompositeFunction] = None
        self._library_numpy: Union[None, tk.Button] = None
        self._library_special: Union[None, tk.Button] = None
        self._library_stats: Union[None, tk.Button] = None
        # self._library_math: tk.Button = None
        self._library_autofit: Union[None, tk.Button] = None
        self._error_label: Union[None, tk.Label] = None
        self._current_name_label: Union[None, tk.Label] = None
        self._current_form_label: Union[None, tk.Label] = None
        self._slider_frame: Union[None, tk.Frame] = None

        # defaults config ------------------------------------------------------------------------->
        self._default_gui_width = 0
        self._default_gui_height = 0
        self._default_gui_x = -10
        self._default_gui_y = -10
        self._default_fit_type = "Linear"
        self._default_excel_x_range: str = "A1:A10"
        self._default_excel_y_range: str = "B1:B10"
        self._default_excel_sigmax_range: str = ""
        self._default_excel_sigmay_range: str = ""
        self._default_load_file_loc: str = ""
        self._default_bg_colour: str = "Default"
        self._default_dataaxes_colour: str = "Default"
        self._default_fit_colour: str = "Default"
        self._default_console_colour: str = "Default"
        self._default_printout_colour: str = "Default"
        self._default_os_scaling: float = 1

        if sys.platform == "darwin":
            self._platform_offset = 4
            self._platform_scale = 1.17
            self._platform_border = 0
            self._right_click = "<Button-2>"
        else:
            self._platform_offset = 0
            self._platform_scale = 0.85
            self._platform_border = 2
            self._right_click = "<Button-3>"

        if sys.platform == "win32":
            self.sym_chi = "\U0001D6D8"
            self.sym_left = "\U0001F844"
            self.sym_up = "\U0001F845"
            self.sym_right = "\U0001F846"
            self.sym_down = "\U0001F847"
            self.sym_sigma = "\U000003C3"
        else:
            self.sym_chi = "\U000003C7"
            self.sym_left = "\U00002190"
            self.sym_up = "\U00002191"
            self.sym_right = "\U00002192"
            self.sym_down = "\U00002193"
            self.sym_sigma = (
                "sigma"  # "\U000003C3"  # u"\U000003C3"  # "\U000003C3".encode('utf-8')
            )
        self.sym_check = " \U00002713"

        if sys.platform == "linux":
            self._sbf = "#d9d9d9"
        else:
            self._sbf = "SystemButtonFace"

        self._image_r: float = 1
        self._custom_function_names: str = ""
        self._custom_function_forms: str = ""
        self._default_manual_name: str = "N/A"
        self._default_manual_form: str = "N/A"
        self._custom_function_button: Union[None, tk.Button] = None

        self._criterion = "rchisqr"  # other opts AIC, AICc, BICc, HQIC

        self._background_menu, self._dataaxis_menu, self._fit_colour_menu = (
            None,
            None,
            None,
        )
        self._background_labels = ["Default", "White", "Dark", "Black"]
        self._dataaxis_labels = ["Default", "White"]
        self._fit_colour_labels = ["Default", "White", "Black"]
        self._printout_background_menu, self._printout_menu = None, None
        self._printout_background_labels = ["Default", "White", "Pale"]
        self._printout_labels = ["Default", "White", "Black"]
        self._refit_menu, self._criterion_menu = None, None
        self._refit_labels = ["Always", "With Button"]
        self._criterion_labels = [
            f"Reduced {self.sym_chi}{sup(2)}",
            "AIC",
            "AICc",
            "BIC",
            "HQIC",
        ]
        self._sliders: list[tk.Scale] = []
        self._slider_labels: list[tk.Label] = []
        self._already_queued = False

        # default configs
        self.touch_defaults()  # required for free version
        self.load_defaults()
        self.print_defaults()

        # load in splash screen
        self.load_splash_screen()

        if not Frontend._meipass_flag:
            self.add_message(f"  Directory is {pkg_path()}")
        logger(f"Package directory is {pkg_path()} with MEIPASS={Frontend._meipass_flag}")
        logger(f"Data directory is {Frontend.get_data_path()}")
        self._gui.geometry(
            f"{self._default_gui_width}x{self._default_gui_height}+{self._default_gui_x}"
            f"+{self._default_gui_y}"
        )  # to fix aspect ratio changing in add_message

    @staticmethod
    def set_meipass() -> None:

        try:
            # for pyinstaller with standalone exe/app
            loc = sys._MEIPASS  # pylint: disable=protected-access
        except AttributeError:
            Frontend._meipass_flag = False
        else:
            if "_MEI" not in loc:
                Frontend._meipass_flag = False

    @staticmethod
    def get_data_path():

        try:
            # for pyinstaller with standalone exe/app
            loc = sys._MEIPASS  # pylint: disable=protected-access
        except AttributeError:
            filepath = ospath.abspath(__file__)
            loc = ospath.dirname(filepath)

        fallback = loc
        failsafe = 0
        while loc[-7:] != "autofit":
            failsafe += 1
            loc = ospath.dirname(loc)
            if loc == ospath.dirname(loc):
                loc = fallback
                break
            if failsafe > 50:
                loc = fallback
                break

        if ospath.exists(f"{loc}/data"):
            loc = loc + "/data"

        return loc

    def touch_defaults(self):
        try:
            with open(f"{pkg_path()}/frontend.cfg", mode="r", encoding="utf-8") as _:
                return
        except FileNotFoundError:
            with open(f"{pkg_path()}/frontend.cfg", mode="a+", encoding="utf-8") as _:
                pass
            self.save_defaults()

    def load_defaults(self):
        with open(f"{pkg_path()}/frontend.cfg", mode="r", encoding="utf-8") as file:
            for line in file:
                if "#GUI_WIDTH" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "1"
                    self._default_gui_width = int(arg)
                    # line = next(file,"")  # could provide a speedup if this is the launch holdup
                elif "#GUI_HEIGHT" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "1"
                    self._default_gui_height = int(arg)
                elif "#GUI_X" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._default_gui_x = int(arg)
                elif "#GUI_Y" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._default_gui_y = int(arg)
                elif "#FIT_TYPE" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "Linear"
                    self._default_fit_type = arg
                elif "#PROCEDURAL_DEPTH" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "3"
                    self._max_functions_tkint.set(arg)
                elif "#EXCEL_RANGE_X" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "A3:A18"
                    self._default_excel_x_range = arg
                elif "#EXCEL_RANGE_Y" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = ""
                    self._default_excel_y_range = arg
                elif "#EXCEL_RANGE_SIGMA_X" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = ""
                    self._default_excel_sigmax_range = arg
                elif "#EXCEL_RANGE_SIGMA_Y" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = ""
                    self._default_excel_sigmay_range = arg
                elif "#LOAD_FILE_LOC" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = f"{Frontend.get_data_path()}"
                    self._default_load_file_loc = arg
                elif "#BG_COLOUR" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "Default"
                    if arg in ["Default", "Black", "White", "Dark"]:
                        self._default_bg_colour = arg
                    else:
                        self._default_bg_colour = "Default"
                    if arg == "Black":
                        self._bg_color = (0.0, 0.0, 0.0)
                    elif arg == "White":
                        self._bg_color = (1.0, 1.0, 1.0)
                    elif arg == "Dark":
                        self._bg_color = (0.2, 0.2, 0.2)
                    else:
                        self._bg_color = (112 / 255, 146 / 255, 190 / 255)
                elif "#DATAAXES_COLOUR" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "Default"

                    if self._default_bg_colour == "White" and arg == "White":
                        # prevent data from also being white
                        arg = "Default"
                    if self._default_bg_colour == "Black" and arg == "Default":
                        # prevent data from also being black
                        arg = "White"

                    if arg in ["Default", "White"]:
                        self._default_dataaxes_colour = arg
                    else:
                        self._default_dataaxes_colour = "Default"
                    if arg == "White":
                        self._dataaxes_color = (1.0, 1.0, 1.0)
                    else:
                        self._dataaxes_color = (0.0, 0.0, 0.0)
                elif "#FIT_COLOUR" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "Default"

                    if self._default_bg_colour == "White" and arg == "White":
                        # prevent fit from also being white
                        arg = "Default"
                    if self._default_bg_colour == "Black" and arg == "Black":
                        # prevent fit from also being black
                        arg = "White"

                    if arg in ["Default", "Black", "White"]:
                        self._default_fit_colour = arg
                    else:
                        self._default_fit_colour = "Default"
                    if arg == "Black":
                        self._fit_color = (0.0, 0.0, 0.0)
                    elif arg == "White":
                        self._fit_color = (1.0, 1.0, 1.0)
                    else:
                        self._fit_color = (1.0, 0.0, 0.0)
                elif "#CONSOLE_COLOUR" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "Default"
                    if arg in ["Default", "White", "Pale"]:
                        self._default_console_colour = arg
                    else:
                        self._default_console_colour = "Default"
                    if arg == "Pale":
                        self._console_color = self._sbf
                    elif arg == "White":
                        self._console_color = "white"
                    else:
                        self._console_color = "black"
                elif "#PRINTOUT_COLOUR" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "Default"
                    if arg in ["Default", "White", "Black"]:
                        self._default_printout_colour = arg
                    else:
                        self._default_printout_colour = "Default"
                    if arg == "White":
                        self._printout_color = (255, 255, 255)
                    elif arg == "Black":
                        self._printout_color = (0, 0, 0)
                    else:
                        self._printout_color = (0, 200, 0)
                elif "#COS_ON" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._use_func_dict_name_tkbool["cos(x)"].set(bool(int(arg)))
                elif "#SIN_ON" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._use_func_dict_name_tkbool["sin(x)"].set(bool(int(arg)))
                elif "#EXP_ON" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._use_func_dict_name_tkbool["exp(x)"].set(bool(int(arg)))
                elif "#LOG_ON" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._use_func_dict_name_tkbool["log(x)"].set(bool(int(arg)))
                elif "#POW_NEG1_ON" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._use_func_dict_name_tkbool["1/x"].set(bool(int(arg)))
                elif "#CUSTOM_ON" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "0"
                    self._use_func_dict_name_tkbool["custom"].set(bool(int(arg)))
                elif "#CUSTOM_NAMES" in line:
                    args = regex.split(" ", line.rstrip("\n \t"))
                    if args == "" or args[0] == "#":
                        self._custom_function_names = ""
                    else:
                        self._custom_function_names = " ".join(args[1:])
                elif "#CUSTOM_FORMS" in line:
                    args = regex.split(" ", line.rstrip("\n \t"))
                    if args == "" or args[0] == "#":
                        self._custom_function_forms = ""
                    else:
                        self._custom_function_forms = " ".join(args[1:])
                elif "#MANUAL_NAME" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "N/A"
                    self._default_manual_name = arg
                elif "#MANUAL_FORM" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "N/A"
                    self._default_manual_form = arg
                elif "#OS_SCALING" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = 1.0
                    self._default_os_scaling = max(float(arg), 0.1)
                elif "#IMAGE_R" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = 1.0
                    self._image_r = min(max(float(arg), 0.1), 10)
                elif "#REFIT_ALWAYS" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "1"
                    if arg in ["0", "1"]:
                        self._refit_on_click = bool(int(arg))
                    else:
                        self._refit_on_click = False
                elif "#CRITERION" in line:
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "rchisqr"
                    if arg in ["rchisqr", "AIC", "AICc", "BIC", "HQIC"]:
                        self.criterion = arg
                    else:
                        self.criterion = "rchisqr"

    def save_defaults(self):
        # logger(f"SAVED DEFAULTS}")
        if self.brute_forcing or self._default_fit_type == "Brute-Force":
            return
        with open(f"{pkg_path()}/frontend.cfg", "w", encoding="utf-8") as file:
            file.write(f"#GUI_WIDTH {self._gui.winfo_width()}\n")
            file.write(f"#GUI_HEIGHT {self._gui.winfo_height()}\n")
            file.write(f"#GUI_X {self._gui.winfo_x()}\n")
            file.write(f"#GUI_Y {self._gui.winfo_y()}\n")
            file.write(f"#FIT_TYPE {self._default_fit_type}\n")
            file.write(f"#PROCEDURAL_DEPTH {self.max_functions}\n")
            file.write(f"#EXCEL_RANGE_X {self._default_excel_x_range}\n")
            file.write(f"#EXCEL_RANGE_Y {self._default_excel_y_range}\n")
            file.write(f"#EXCEL_RANGE_SIGMA_X {self._default_excel_sigmax_range}\n")
            file.write(f"#EXCEL_RANGE_SIGMA_Y {self._default_excel_sigmay_range}\n")
            file.write(f"#LOAD_FILE_LOC {self._default_load_file_loc}\n")
            file.write(f"#BG_COLOUR {self._default_bg_colour}\n")
            file.write(f"#DATAAXES_COLOUR {self._default_dataaxes_colour}\n")
            file.write(f"#FIT_COLOUR {self._default_fit_colour}\n")
            file.write(f"#CONSOLE_COLOUR {self._default_console_colour}\n")
            file.write(f"#PRINTOUT_COLOUR {self._default_printout_colour}\n")
            cos_on = int(self._use_func_dict_name_tkbool["cos(x)"].get())
            sin_on = int(self._use_func_dict_name_tkbool["sin(x)"].get())
            exp_on = int(self._use_func_dict_name_tkbool["exp(x)"].get())
            log_on = int(self._use_func_dict_name_tkbool["log(x)"].get())
            pow_neg1_on = int(self._use_func_dict_name_tkbool["1/x"].get())
            custom_on = int(self._use_func_dict_name_tkbool["custom"].get())
            if all([cos_on, sin_on, exp_on, log_on, pow_neg1_on, custom_on]):
                logger(
                    "You shouldn't have all functions turned on for a procedural fit. "
                    "Use brute-force instead."
                )
                logger(f" {self.brute_forcing=} {self._default_fit_type=}")
            file.write(f"#COS_ON {cos_on}\n")
            file.write(f"#SIN_ON {sin_on}\n")
            file.write(f"#EXP_ON {exp_on}\n")
            file.write(f"#LOG_ON {log_on}\n")
            file.write(f"#POW_NEG1_ON {pow_neg1_on}\n")
            file.write(f"#CUSTOM_ON {custom_on}\n")
            file.write(f"#CUSTOM_NAMES {self._custom_function_names}\n")
            file.write(f"#CUSTOM_FORMS {self._custom_function_forms}\n")
            file.write(f"#MANUAL_NAME {self._default_manual_name}\n")
            file.write(f"#MANUAL_FORM {self._default_manual_form}\n")
            file.write(f"#OS_SCALING {self._default_os_scaling}\n")
            file.write(f"#IMAGE_R {self._image_r}\n")
            file.write(f"#REFIT_ALWAYS {1 if self._refit_on_click else 0}\n")
            file.write(f"#CRITERION {self.criterion}\n")

    def print_defaults(self):
        logger(f"GUI Width >{self._default_gui_width}<")
        logger(f"GUI Height >{self._default_gui_height}<")
        logger(f"GUI X >{self._default_gui_x}<")
        logger(f"GUI Y >{self._default_gui_y}<")
        logger(f"Fit-type >{self._default_fit_type}<")
        logger(f"Procedural depth >{self.max_functions}<")
        logger(f"Excel X-Range >{self._default_excel_x_range}<")
        logger(f"Excel Y-Range >{self._default_excel_y_range}<")
        logger(f"Excel SigmaX-Range >{self._default_excel_sigmax_range}<")
        logger(f"Excel SigmaY-Range >{self._default_excel_sigmay_range}<")
        logger(f"Data location >{self._default_load_file_loc}<")
        logger(f"Background Colour >{self._default_bg_colour}<")
        logger(f"Data and Axis Colour >{self._default_dataaxes_colour}<")
        logger(f"Fit Line Colour >{self._default_fit_colour}<")
        logger(f"Console Colour >{self._default_console_colour}<")
        logger(f"Printout Colour >{self._default_printout_colour}<")
        cos_on = int(self._use_func_dict_name_tkbool["cos(x)"].get())
        sin_on = int(self._use_func_dict_name_tkbool["sin(x)"].get())
        exp_on = int(self._use_func_dict_name_tkbool["exp(x)"].get())
        log_on = int(self._use_func_dict_name_tkbool["log(x)"].get())
        pow_neg1_on = int(self._use_func_dict_name_tkbool["1/x"].get())
        custom_on = int(self._use_func_dict_name_tkbool["custom"].get())
        logger(f"Procedural cos(x) >{cos_on}<")
        logger(f"Procedural sin(x) >{sin_on}<")
        logger(f"Procedural exp(x) >{exp_on}<")
        logger(f"Procedural log(x) >{log_on}<")
        logger(f"Procedural 1/x >{pow_neg1_on}<")
        logger(f"Procedural custom >{custom_on}<")
        logger(f"Custom function names >{self._custom_function_names}<")
        logger(f"Custom function forms >{self._custom_function_forms}<")
        logger(f"Manual function name >{self._default_manual_name}<")
        logger(f"Manual function form >{self._default_manual_form}<")
        logger(f"OS Scaling >{self._default_os_scaling:.2F}<")
        logger(f"Image R >{self._image_r:.3F}<")
        logger(f"Refit on Click: >{1 if self._refit_on_click else 0}<")
        logger(f"Criterion: >{self.criterion}<")

    # create left, right, and middle panels
    def load_splash_screen(self):

        gui = self._gui

        # window size and title
        if self._default_gui_width <= self._os_width / 4 + 1:
            logger(f"Undersized width {self._default_gui_width} {self._os_width}")
            self._default_gui_width = self._os_width * 3 // 4
        else:
            logger(f"Fine width {self._default_gui_width} {self._os_width * 7 / 6}")
            self._default_gui_width = min(self._default_gui_width, self._os_width * 7 // 6)

        if self._default_gui_height <= self._os_height / 4 + 1:
            logger(f"Undersized height {self._default_gui_height} {self._os_height}")
            self._default_gui_height = self._os_height * 3 // 4
        else:
            logger(f"Fine height {self._default_gui_height} {self._os_height * 7 / 6}")
            self._default_gui_height = min(self._default_gui_height, self._os_height * 7 // 6)

        gui.geometry(
            f"{self._default_gui_width}x{self._default_gui_height}"
            f"+{self._default_gui_x}+{self._default_gui_y}"
        )
        gui.rowconfigure(0, minsize=400, weight=1)

        # icon image and window title
        if sys.platform == "win32":
            gui.iconbitmap(f"{pkg_path()}/images/icon.ico")
        elif sys.platform == "darwin":
            photo = tk.PhotoImage(file=f"{pkg_path()}/images/splash.png")
            gui.wm_iconphoto(False, photo)
        elif sys.platform == "linux":
            iconify = tk.PhotoImage(file=f"{pkg_path()}/images/splash.png")
            gui.wm_iconphoto(False, iconify)
        gui.title("MIW's AutoFit")

        # menus
        self.create_file_menu()

        # left panel -- menu buttons
        self.create_left_panel()
        # middle panel -- data visualization, fit options, data transforms
        self.create_middle_panel()
        # right panel -- text output
        self.create_right_panel(hello_str="> Welcome to MIW's AutoFit!")

    # MENUS
    def create_file_menu(self):

        menu_bar = tk.Menu(self._gui)

        self._gui.config(menu=menu_bar)

        # FILE menu
        if sys.platform != "darwin":
            file_menu = tk.Menu(master=menu_bar, tearoff=0)
            menu_bar.add_cascade(label="File", menu=file_menu, underline=0)

            file_menu.add_command(label="Open", command=self.load_data_command)

            restart_menu = tk.Menu(master=file_menu, tearoff=0)
            restart_are_you_sure_menu = tk.Menu(master=restart_menu, tearoff=0)
            restart_are_you_sure_menu.add_command(label="Yes", command=self.restart_command)

            file_menu.add_cascade(label="Restart", menu=restart_menu)
            restart_menu.add_cascade(label="Are you sure?", menu=restart_are_you_sure_menu)

            exit_menu = tk.Menu(master=file_menu, tearoff=0)
            exit_are_you_sure_menu = tk.Menu(master=exit_menu, tearoff=0)
            exit_are_you_sure_menu.add_command(label="Yes", command=self._gui.destroy)

            file_menu.add_cascade(label="Exit", menu=exit_menu)
            exit_menu.add_cascade(label="Are you sure?", menu=exit_are_you_sure_menu)

        # SETTINGS menu
        settings_menu = tk.Menu(master=menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Settings", menu=settings_menu, underline=0)

        # appearance
        appearance_menu = tk.Menu(master=settings_menu, tearoff=0)

        self._background_menu = tk.Menu(master=appearance_menu, tearoff=0)
        self._background_menu.add_command(label="Default", command=self.bg_color_default)
        self._background_menu.add_command(label="White", command=self.bg_color_white)
        self._background_menu.add_command(label="Dark", command=self.bg_color_dark)
        self._background_menu.add_command(label="Black", command=self.bg_color_black)
        self.checkmark_background_options(self._background_labels.index(self._default_bg_colour))

        self._dataaxis_menu = tk.Menu(master=appearance_menu, tearoff=0)
        self._dataaxis_menu.add_command(label="Default", command=self.dataaxes_color_default)
        self._dataaxis_menu.add_command(label="White", command=self.dataaxes_color_white)
        self.checkmark_dataaxis_options(self._dataaxis_labels.index(self._default_dataaxes_colour))

        self._fit_colour_menu = tk.Menu(master=appearance_menu, tearoff=0)
        self._fit_colour_menu.add_command(label="Default", command=self.fit_color_default)
        self._fit_colour_menu.add_command(label="White", command=self.fit_color_white)
        self._fit_colour_menu.add_command(label="Black", command=self.fit_color_black)
        self.checkmark_fit_colour_options(self._fit_colour_labels.index(self._default_fit_colour))

        image_size_menu = tk.Menu(master=appearance_menu, tearoff=0)
        image_size_menu.add_command(label="Up", command=self.mouse_wheel_up)
        image_size_menu.add_command(label="Down", command=self.mouse_wheel_down)

        self._printout_background_menu = tk.Menu(master=appearance_menu, tearoff=0)
        self._printout_background_menu.add_command(
            label="Default", command=self.console_color_default
        )
        self._printout_background_menu.add_command(label="White", command=self.console_color_white)
        self._printout_background_menu.add_command(label="Pale", command=self.console_color_pale)
        self.checkmark_printout_background_options(
            self._printout_background_labels.index(self._default_console_colour)
        )

        self._printout_menu = tk.Menu(master=appearance_menu, tearoff=0)
        self._printout_menu.add_command(label="Default", command=self.printout_color_default)
        self._printout_menu.add_command(label="White", command=self.printout_color_white)
        self._printout_menu.add_command(label="Black", command=self.printout_color_black)
        self.checkmark_printout_options(self._printout_labels.index(self._default_printout_colour))

        gui_resolution_menu = tk.Menu(master=appearance_menu, tearoff=0)
        gui_resolution_menu.add_command(label="Up", command=self.size_up)
        gui_resolution_menu.add_command(label="Down", command=self.size_down)

        settings_menu.add_cascade(label="Appearance", menu=appearance_menu)
        appearance_menu.add_cascade(label="Image Background", menu=self._background_menu)
        appearance_menu.add_cascade(label="Data/Axis Colour", menu=self._dataaxis_menu)
        appearance_menu.add_cascade(label="Fit Colour", menu=self._fit_colour_menu)
        appearance_menu.add_cascade(label="Image Size", menu=image_size_menu)
        appearance_menu.add_cascade(
            label="Printout Background", menu=self._printout_background_menu
        )
        appearance_menu.add_cascade(label="Printout Colour", menu=self._printout_menu)
        appearance_menu.add_cascade(label="Text Size", menu=gui_resolution_menu)

        # behaviour
        behaviour_menu = tk.Menu(master=settings_menu, tearoff=0)

        self._refit_menu = tk.Menu(master=behaviour_menu, tearoff=0)
        self._refit_menu.add_command(label="Always", command=self.refit_always)
        self._refit_menu.add_command(label="With Button", command=self.refit_sometimes)
        self.checkmark_refit_options(
            0 if self._refit_on_click else 1
        )  # 0 is the index of always, 1 of with button

        self._criterion_menu = tk.Menu(master=behaviour_menu, tearoff=0)
        self._criterion_menu.add_command(
            label=f"Reduced {self.sym_chi}{sup(2)}", command=self.criterion_rchisqr
        )
        self._criterion_menu.add_command(label="AIC", command=self.criterion_AIC)
        self._criterion_menu.add_command(label="AICc", command=self.criterion_AICc)
        self._criterion_menu.add_command(label="BIC", command=self.criterion_BIC)
        self._criterion_menu.add_command(label="HQIC", command=self.criterion_HQIC)
        self.checkmark_criterion_options(
            self._criterion_labels.index(self._criterion) if self._criterion != "rchisqr" else 0
        )

        settings_menu.add_cascade(label="Behaviour", menu=behaviour_menu)
        behaviour_menu.add_cascade(label="Refit?", menu=self._refit_menu)
        behaviour_menu.add_cascade(label="Criterions", menu=self._criterion_menu)

    # def create_tutorial_menu(self):
    #     pass

    # Panels
    def create_left_panel(self):
        self._left_panel_frame = tk.Frame(
            master=self._gui, relief=tk.RAISED, bg="white", height=self._os_height
        )
        self._left_panel_frame.grid(row=0, column=0, sticky="ns")
        # logger("Left panel:",self._left_panel_frame.winfo_height(), self._os_height)
        self.create_load_data_button()

    def create_middle_panel(self):
        self._gui.columnconfigure(1, minsize=128)  # image panel
        self._middle_panel_frame = tk.Frame(master=self._gui, bg=self._sbf)
        self._middle_panel_frame.grid(row=0, column=1, sticky="nsew")
        self.create_image_frame()  # aka image frame
        self.create_data_perusal_frame()  # aka inspect frame
        self.create_fit_options_frame()  # aka dropdown/checkbox frame
        self.create_plot_options_frame()  # aka log normalize frame
        self.create_procedural_frame()  # aka procedural options frame
        self.create_polynomial_frame()  # aka polynomial frame
        self.create_gaussian_frame()  # aka gaussian frame
        self.create_manual_frame()

    def create_right_panel(self, hello_str=""):
        self._gui.columnconfigure(2, minsize=128, weight=1)  # image panel
        self._right_panel_frame = tk.Frame(master=self._gui, bg=self._console_color)
        self._right_panel_frame.grid(row=0, column=2, sticky="news")
        self.add_message(hello_str)
        self._right_panel_frame.bind(self._right_click, self.do_colors_console_popup)

    # LEFT PANEL FUNCTIONS ------------------------------------------------------------------------>

    def create_load_data_button(self):
        load_data_button = tk.Button(
            master=self._gui.children["!frame"],
            text="Load Data",
            width=len("Load Data") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.load_data_command,
        )
        load_data_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    def load_data_command(self):

        new_filepaths = list(
            fd.askopenfilenames(
                parent=self._gui,
                initialdir=self._default_load_file_loc,
                title="Select a file to fit",
                filetypes=(
                    ("All Files", "*.*"),
                    ("Comma-Separated Files", "*.csv *.txt"),
                    ("Spreadsheets", "*.xls *.ods"),
                ),
            )
        )
        # trim duplicates
        for path in new_filepaths[:]:
            if path in self._filepaths:
                shortpath = regex.split("/", path)[-1]
                logger(f"{shortpath} already loaded")
                new_filepaths.remove(path)
            elif path[-4:] == "xlsx":
                if sys.platform in ["darwin", "linux"]:
                    self.add_message("\n \n> .xlsx file format not supported.")
                    new_filepaths.remove(path)
        for path in new_filepaths[:]:
            if path[-4:] in [".xls", ".ods", "xlsx"] and self._new_user_stage % 23 != 0:
                self.dialog_box_get_excel_data_ranges()
                logger(f"{self._excel_x_range=} {self._excel_y_range=}")
                if self._excel_x_range == "":
                    # the user didn't actually want to load that file
                    new_filepaths.remove(path)
                    continue
                self._new_user_stage *= 23
                sheet_names = ExcelFile(path).sheet_names
                if self._all_sheets_in_file.get():
                    for _ in range(len(sheet_names) - 1):
                        self._filepaths.append(path)
                logger(f"In this file the sheets names are {sheet_names}")
            self._default_load_file_loc = "/".join(regex.split("/", path)[:-1])
            self._filepaths.append(path)
            # self._normalized_histogram_flags.append(False)

        if len(new_filepaths) == 0:
            return

        if self.brute_forcing or self._default_fit_type == "Brute Force":
            logger("In load data command, we're loading a file while brute-forcing is on")
            self.brute_forcing = False

        if self._new_user_stage % 2 != 0:
            self.create_fit_button()
            self._new_user_stage *= 2

        # add buttons to adjust fit options
        if self._default_fit_type != "Linear":
            self.show_function_dropdown()
        # degree options for polynomial fits
        if self._model_name_tkstr.get() == "Polynomial":
            self.show_degree_buttons()
        elif self._model_name_tkstr.get() == "Gaussian":
            self.show_modal_buttons()
        # checkbox and depth options for procedural fits
        elif self._model_name_tkstr.get() == "Procedural":
            self.show_procedural_options()
        elif self._model_name_tkstr.get() == "Manual":
            self.show_manual_fields()

        if len(new_filepaths) > 0:
            self._changed_data_flag = True
            self._curr_image_num = len(self._data_handlers)
            self.load_new_data(new_filepaths)
            if self._showing_fit_image:
                if self._refit_on_click:
                    # this fits the new data -- make this an option?
                    self.show_current_data_with_fit()
                else:
                    # this shouldn't reprint the model, since we arent refitting
                    self.save_show_fit_image()
            else:
                self.show_current_data()
            if self._new_user_stage % 3 != 0:
                self.create_inspect_button()
                self._new_user_stage *= 3
            logger(f"Loaded {len(new_filepaths)} files.")

        # update dropdown with new chi_sqrs for the current top 5 models,
        # but according to the original parameters
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()
            logger("If refit on button, this should make refit_button appear")
            if len(self._data_handlers) > 1:
                logger("If refit on button, this should make refit_button appear")
                self.show_refit_button()

        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force", "Manual"]:
            self.show_custom_function_button()
        else:
            self.hide_custom_function_button()

        logger(self._filepaths)
        if len(self._filepaths) > 1:
            self.show_left_right_buttons()
            self.update_data_select()

            if self._showing_fit_image:
                self.create_fit_all_button()

        self.update_logx_relief()
        self.update_logy_relief()
        self.save_defaults()

        self.update_optimizer()

    def dialog_box_get_excel_data_ranges(self):

        dialog_box = tk.Toplevel()
        dialog_box.geometry(
            f"{self._image_frame.winfo_width() * 4 // 5}"
            f"x{self._image_frame.winfo_height() * 6 // 10}"
        )
        dialog_box.title("Spreadsheet Input Options")
        if sys.platform == "win32":
            dialog_box.iconbitmap(f"{pkg_path()}/images/icon.ico")
        elif sys.platform == "darwin":
            photo = tk.PhotoImage(file=f"{pkg_path()}/images/splash.png")
            dialog_box.iconphoto(False, photo)
        elif sys.platform == "linux":
            icon = tk.PhotoImage(file=f"{pkg_path()}/images/icon.png")
            dialog_box.iconphoto(False, icon)
        else:
            pass

        data_frame = tk.Frame(master=dialog_box)
        data_frame.grid(row=0, column=0, sticky="ew")
        exp_frame = tk.Frame(master=dialog_box)
        exp_frame.grid(row=1, column=0, sticky="w")

        x_label = tk.Label(master=data_frame, text="Cells for x values: ")
        x_label.grid(row=0, column=0, sticky="w")
        x_data = tk.Entry(master=data_frame)
        x_data.insert(0, self._default_excel_x_range)
        x_data.grid(row=0, column=1, sticky="w")

        y_label = tk.Label(master=data_frame, text="Cells for y values: ")
        y_label.grid(row=1, column=0, sticky="w")
        y_data = tk.Entry(master=data_frame)
        y_data.insert(0, self._default_excel_y_range)
        y_data.grid(row=1, column=1, sticky="w")

        sigmax_label = tk.Label(master=data_frame, text="Cells for x uncertainties: ")
        sigmax_label.grid(row=2, column=0, sticky="w")
        sigmax_data = tk.Entry(master=data_frame)
        sigmax_data.insert(0, self._default_excel_sigmax_range)
        sigmax_data.grid(row=2, column=1, sticky="w")

        sigmay_label = tk.Label(master=data_frame, text="Cells for y uncertainties: ")
        sigmay_label.grid(row=3, column=0, sticky="w")
        sigmay_data = tk.Entry(master=data_frame)
        sigmay_data.insert(0, self._default_excel_sigmay_range)
        sigmay_data.grid(row=3, column=1, sticky="w")

        checkbox = tk.Checkbutton(
            master=data_frame,
            text="Range applies to all sheets in the file",
            variable=self._all_sheets_in_file,
            onvalue=True,
            offvalue=False,
        )
        checkbox.grid(row=4, column=0, sticky="w")

        example_label = tk.Label(master=data_frame, text="\nFormatting Example")
        example_label.grid(row=10, column=0, sticky="w")
        example_data = tk.Entry(master=data_frame)
        example_data.insert(0, "D1:D51")
        example_data.grid(row=10, column=1, sticky="ws")

        close_dialog_button = tk.Button(
            master=data_frame,
            text="OK",
            width=len("OK") - self._platform_offset,
            # font=('TkDefaultFont', int(12*self._default_os_scaling*self._platform_scale)),
            bd=self._platform_border,
            command=self.close_dialog_box_command_excel,
        )
        close_dialog_button.grid(row=0, column=10, padx=5, pady=0, sticky="ns")
        dialog_box.bind("<Return>", self.close_dialog_box_command_excel)
        dialog_box.focus_force()

        if sys.platform == "win32":
            explanation_label = tk.Label(
                master=exp_frame,
                text="\nThese settings will apply to all " ".xls, .xlsx, and .ods files",
            )
        else:
            explanation_label = tk.Label(
                master=exp_frame,
                text="\nThese settings will apply to all " ".xls and .ods files",
            )
        explanation_label.grid(row=0, column=0, sticky="w")

        self._popup_window = dialog_box
        self._gui.wait_window(dialog_box)

    # noinspection PyUnusedLocal
    def close_dialog_box_command_excel(self, bind_command=None):  # pylint: disable=unused-argument

        if self._popup_window is None:
            logger("Window already closed")
        self._excel_x_range = self._popup_window.children["!frame"].children["!entry"].get()
        self._excel_y_range = self._popup_window.children["!frame"].children["!entry2"].get()
        self._excel_sigmax_range = self._popup_window.children["!frame"].children["!entry3"].get()
        self._excel_sigmay_range = self._popup_window.children["!frame"].children["!entry4"].get()

        if not DataHandler.valid_excel_endpoints(self._excel_x_range):
            self.add_message(f"\n \n> Invalid x-range {self._excel_x_range}")
            return
        if not DataHandler.valid_excel_endpoints(self._excel_y_range):
            self.add_message(f"\n \n> Invalid y-range {self._excel_y_range}")
            return
        if not DataHandler.valid_excel_endpoints(self._excel_sigmax_range):
            self.add_message(f"\n \n> Invalid range for x uncertainties {self._excel_sigmax_range}")
            return
        if not DataHandler.valid_excel_endpoints(self._excel_sigmay_range):
            self.add_message(f"\n \n> Invalid range for y uncertainties {self._excel_sigmay_range}")
            return

        self._default_excel_x_range = self._excel_x_range
        self._default_excel_y_range = self._excel_y_range
        self._default_excel_sigmax_range = self._excel_sigmax_range
        self._default_excel_sigmay_range = self._excel_sigmay_range

        self.save_defaults()
        self._popup_window.destroy()

    # Helper functions
    def load_new_data(self, new_filepaths_lists):
        for path in new_filepaths_lists:
            if path[-4:] in [".xls", "xlsx", ".ods"]:
                for sheet_name in ExcelFile(path).sheet_names:
                    self._data_handlers.append(DataHandler(filepath=path))
                    self._data_handlers[-1].set_excel_sheet_name(sheet_name)
                    self._data_handlers[-1].set_excel_args(
                        x_range_str=self._excel_x_range,
                        y_range_str=self._excel_y_range,
                        x_error_str=self._excel_sigmax_range,
                        y_error_str=self._excel_sigmay_range,
                    )
                    if not self._all_sheets_in_file.get():
                        break

            else:
                # only add one data handler
                self._data_handlers.append(DataHandler(filepath=path))

    def show_data(self):

        self._image_path = f"{pkg_path()}/plots/front_end_current_plot.png"
        # create a scatter plot of the first file

        x_points = self.data_handler.unlogged_x_data
        y_points = self.data_handler.unlogged_y_data
        sigma_x_points = self.data_handler.unlogged_sigmax_data
        sigma_y_points = self.data_handler.unlogged_sigmay_data

        plt.close()
        plt.figure(
            facecolor=self._bg_color,
            figsize=(6.4 * self._image_r, 4.8 * self._image_r),
            dpi=100 + int(np.log10(len(x_points))),
        )

        plt.errorbar(
            x_points,
            y_points,
            xerr=sigma_x_points,
            yerr=sigma_y_points,
            fmt="o",
            color=self._dataaxes_color,
        )
        plt.xlabel(self.data_handler.x_label)
        plt.ylabel(self.data_handler.y_label)
        axes = plt.gca()
        axes.tick_params(color=self._dataaxes_color, labelcolor=self._dataaxes_color)
        axes.xaxis.label.set_color(self._dataaxes_color)
        axes.yaxis.label.set_color(self._dataaxes_color)
        for spine in axes.spines.values():
            spine.set_edgecolor(self._dataaxes_color)

        pf.zero_out_axes(axes)

        if self.data_handler.logx_flag:
            logger("Setting log x-scale in show_data")
            pf.set_xaxis_format_log(axes, x_points)
        else:
            pf.set_xaxis_format_linear(axes, x_points)

        if self.data_handler.logy_flag:
            logger("Setting log y-scale in show_data")
            pf.set_yaxis_format_log(axes, y_points)
        else:
            pf.set_yaxis_format_linear(axes, y_points)

        axes.set_facecolor(self._bg_color)

        pf.fix_axes_labels(
            axes,
            min(x_points),
            max(x_points),
            min(y_points),
            max(y_points),
            self.data_handler.x_label,
        )
        plt.savefig(self._image_path, facecolor=self._bg_color)

        # replace the splash graphic with the plot
        self.switch_image()

        # if we're showing the image, we want the optimizer to be working with this data
        if self._showing_fit_image:
            data = self.data_handler.data
            self._optimizer.set_data_to(data)

        # add logx and logy to plot options frame
        self.show_log_buttons()

        # if it's a histogram, add an option to normalize the data to the plot options frame
        if self.data_handler.histogram_flag:
            # normalize behaves... badly and confusingly with logged data
            if not self.data_handler.logx_flag and not self.data_handler.logy_flag:
                self.show_normalize_button()
        else:
            self.hide_normalize_button()

    def create_fit_button(self):
        self._fit_data_button = tk.Button(
            master=self._gui.children["!frame"],
            text="Fit Data",
            width=len("Fit Data") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.fit_data_command,
        )
        self._fit_data_button.grid(row=1, column=0, sticky="ew", padx=5)

    def fit_data_command(self):

        if self._fit_data_button["text"] == "Cancel":
            self._progress_label.configure(bg="#010101")
            return

        # add dropdown for option to select different fit models
        self.show_function_dropdown()

        self.update_optimizer()

        # Find the fit for the currently displayed data
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            if self.current_model is not None:
                self.optimizer.query_add_to_top5(self.current_model, self.current_covariance)

        if self._model_name_tkstr.get() == "Linear":
            logger("Fitting to linear model")
            plot_model = CompositeFunction.built_in("Linear")
            self.optimizer.fit_this_and_get_model_and_covariance(plot_model)
        elif self._model_name_tkstr.get() == "Polynomial":
            logger(f"Fitting to polynomial model of degree {self._polynomial_degree_tkint.get()}")
            plot_model = CompositeFunction.built_in(
                f"Polynomial{self._polynomial_degree_tkint.get()}"
            )
            self.optimizer.fit_this_and_get_model_and_covariance(plot_model)
        elif self._model_name_tkstr.get() == "Gaussian" and self.data_handler.normalized:
            if self._gaussian_modal_tkint.get() > 1:
                plot_model = CompositeFunction.built_in(
                    f"Gaussian{self._gaussian_modal_tkint.get()}"
                )
            else:
                plot_model = CompositeFunction.built_in("Normal")
            logger(f"Fitting to {plot_model.name} distribution")
            self.optimizer.fit_this_and_get_model_and_covariance(plot_model)
        elif self._model_name_tkstr.get() == "Gaussian":
            plot_model = CompositeFunction.built_in(f"Gaussian{self._gaussian_modal_tkint.get()}")
            logger(f"Fitting to {plot_model.name} distribution")
            self.optimizer.fit_this_and_get_model_and_covariance(plot_model)
        elif self._model_name_tkstr.get() == "Sigmoid":
            logger("Fitting to Sigmoid model")
            plot_model = CompositeFunction.built_in("Sigmoid")
            self.optimizer.fit_this_and_get_model_and_covariance(plot_model)
        elif self._model_name_tkstr.get() == "Procedural":

            num_on = 2 + sum(
                (1 if tkBool.get() else 0 for tkBool in self._use_func_dict_name_tkbool.values())
            )
            num_nodes = num_on
            num_added = num_on
            for depth in range(self._max_functions_tkint.get() - 1):
                # n_comps * n_nodes_per_comp * (sum+mul) * n_prims
                num_added = num_added * (depth + 2) * 2 * num_on
                num_nodes += num_added
            self.add_message(
                f"\n \n> Fitting to procedural model -- expecting to "
                f"generate ~{num_nodes} naive models."
            )

            self.add_message(f"   Stage 0/3: {0:>10} naive models generated, {0:>10} models fit.")
            self._fit_data_button.configure(text="Cancel")
            self.optimizer.find_best_model_for_dataset(status_bar=self._progress_label)
            self._fit_data_button.configure(text="Fit Data")

            # import pickle as pkl
            # with open('optimizer_pickle.pkl', 'wb') as f_o:
            #     pkl.dump(self.optimizer, f_o)

        elif self._model_name_tkstr.get() == "Brute-Force":
            logger("Brute forcing a procedural model")
            self.brute_forcing = True
            # for name in self._checkbox_names_list:
            #     self._use_func_dict_name_tkVar[name].set(value=True)
            self._use_func_dict_name_tkbool["cos(x)"].set(value=True)
            self._use_func_dict_name_tkbool["sin(x)"].set(value=True)
            self._use_func_dict_name_tkbool["exp(x)"].set(value=True)
            self._use_func_dict_name_tkbool["log(x)"].set(value=True)
            self._use_func_dict_name_tkbool["1/x"].set(value=True)
            self._use_func_dict_name_tkbool["custom"].set(value=True)

            self._changed_optimizer_opts_flag = True
            self.update_optimizer()
            self.optimizer.async_find_best_model_for_dataset(start=True)
        elif self._model_name_tkstr.get() == "Manual":

            if self._manual_model is not None:
                logger(f"Fitting data to {self._manual_model.name} model.")
                try:

                    if self.get_slider_args() and (
                        len(self.get_slider_args()) == self._manual_model.dof
                    ):
                        self.optimizer.fit_this_and_get_model_and_covariance(
                            model_=self._manual_model,
                            initial_guess=self.get_slider_args(),
                        )
                    else:
                        logger(
                            f"Fit data command: manual fit: misaligned slider number"
                            f" {len(self.get_slider_args()) if self.get_slider_args() else 0} "
                            f"and model dof {self._manual_model.dof}"
                        )
                        self.optimizer.fit_this_and_get_model_and_covariance(
                            model_=self._manual_model
                        )
                        self.update_sliders()
                except ValueError:
                    self.add_message(
                        "\n \n> It is likely that the domain of your manual function\n"
                        "  is incompatible with the data."
                    )
                    return
            else:
                self.add_message("\n \n> You must validate the model before fitting.")
                return
        else:
            logger(f"Invalid model name {self._model_name_tkstr.get()}")

        # add fit all button if there's more than one file
        if len(self._data_handlers) > 1:
            self.create_fit_all_button()

        self.create_residuals_button()
        self.create_error_bands_button()

        # degree changes for Polynomial
        if self._model_name_tkstr.get() == "Polynomial":
            self.show_degree_buttons()
        else:
            self.hide_degree_buttons()

        # modal changes for Gaussian
        if self._model_name_tkstr.get() == "Gaussian":
            self.show_modal_buttons()
        else:
            self.hide_modal_buttons()

        # add a dropdown list for procedural-type fits
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.make_top_shown()
            self.update_top5_dropdown()
            self.show_top5_dropdown()
            if len(self._data_handlers) > 1:
                self.show_refit_button()
        else:
            self.hide_top5_dropdown()
            self.hide_refit_button()

        # checkbox, depth options, and custom function for procedural fits
        if self._model_name_tkstr.get() == "Procedural":
            self.show_procedural_options()
        else:
            self.hide_procedural_options()

        # brute-force conditionals
        if self._model_name_tkstr.get() == "Brute-Force":
            self.show_pause_button()
        else:
            self.hide_pause_button()
            # print out the parameters on the right
            self.add_message(f"\n \n> For {self.data_handler.shortpath} \n")
            self.print_results_to_console()

        if self._model_name_tkstr.get() == "Manual":
            self.show_manual_fields()
            self.create_sliders()
        else:
            self.hide_manual_fields()

        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force", "Manual"]:
            self.show_custom_function_button()
        else:
            self.hide_custom_function_button()

        self.save_show_fit_image()
        self._default_fit_type = self._model_name_tkstr.get()
        self.save_defaults()

        if self.brute_forcing:
            self.begin_brute_loop()

    def create_fit_all_button(self):
        if self._new_user_stage % 11 == 0:
            return
        self._new_user_stage *= 11

        load_data_button = tk.Button(
            master=self._gui.children["!frame"],
            text="Fit All",
            width=len("Fit All") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.fit_all_command,
        )
        load_data_button.grid(row=2, column=0, sticky="new", padx=5)

    def fit_all_command(self, quiet=False):

        # self.add_message("\n \n> Fitting all datasets\n")

        # need to log all datasets if the current one is logged, and unlog if they ARE logged
        for handler in self._data_handlers:
            if handler is self.data_handler:
                continue

            if self.data_handler.logx_flag:
                if handler.logx_flag:
                    # unlog then relog
                    handler.logx_flag = False
                handler.X0 = -self.data_handler.X0  # links the two X0 values
                handler.logx_flag = True
                if not handler.logx_flag:
                    self.add_message(
                        f"\n \n> Can't log the x-data for {handler.shortpath}. Fit All failed."
                    )
                    return
            elif not self.data_handler.logx_flag and handler.logx_flag:
                handler.logx_flag = False

            if self.data_handler.logy_flag:
                if handler.logy_flag:
                    # unlog then relog
                    handler.logy_flag = False
                handler.Y0 = -self.data_handler.Y0  # links the two Y0 values
                handler.logy_flag = True
                if not handler.logy_flag:
                    self.add_message(
                        f"\n \n> Can't log the y-data for {handler.shortpath}. Fit All failed."
                    )
                    return
            elif not self.data_handler.logy_flag and handler.logy_flag:
                handler.logy_flag = False

        # need to normalize all datasets if the current one is normalized
        if (
            any((handler.normalized for handler in self._data_handlers))
            and not self.data_handler.normalized
        ):
            status_good = self.data_handler.normalize_histogram_data(error_handler=self.add_message)
            if not status_good:
                return
        for handler in self._data_handlers:
            if self.data_handler.normalized and not handler.normalized:
                status_good = handler.normalize_histogram_data()
                if not status_good:
                    return

        # fit every loaded dataset with the current model and return the average parameters
        list_of_args = []
        list_of_uncertainties = []
        for handler in self._data_handlers:
            data = handler.data
            self.optimizer.set_data_to(data)
            # does the following line actually use the chosen model?
            self.optimizer.fit_this_and_get_model_and_covariance(
                model_=self.current_model,
                initial_guess=self.current_model.args,
                do_halving=True,
            )
            list_of_args.append(self.current_args)
            logger(f"Beelzebub={handler.shortpath} {self.current_args} +- {self.current_uncs}")
            list_of_uncertainties.append(self.current_uncs)

        means = []
        uncs = []
        N = len(list_of_args)
        for idx, _ in enumerate(list_of_args[0]):
            sum_args = 0
            for par_list in list_of_args:
                sum_args += par_list[idx]
            mean = sum_args / N

            sum_uncertainty_sqr = 0
            sum_variance = 0
            for par_list, unc_list in zip(list_of_args, list_of_uncertainties):
                sum_uncertainty_sqr += unc_list[idx] ** 2 / N**2
                sum_variance += (par_list[idx] - mean) ** 2 / (N - 1) if N > 1 else 0

            ratio = sum_variance / (sum_variance + sum_uncertainty_sqr)
            effective_variance = ratio * sum_variance + (1 - ratio) * sum_uncertainty_sqr

            means.append(mean)
            uncs.append(np.sqrt(effective_variance))

        logger(f"{means} +- {uncs}")
        fit_all_model = self.current_model.copy()
        fit_all_model.args = means
        self.optimizer.shown_model = fit_all_model
        a = np.zeros((len(means), len(means)))
        np.fill_diagonal(a, uncs)
        self.optimizer.shown_covariance = a

        self.save_show_fit_all(args_list=list_of_args)

        if not quiet:
            self.add_message("\n \n> Average parameters from fitting all datasets:\n")
            self.print_results_to_console()
        self.update_data_select()

    def create_custom_function_button(self):
        if self._new_user_stage % 41 == 0:
            return
        self._new_user_stage *= 41
        left_panel_bottom = tk.Frame(self._gui.children["!frame"], bg="white")
        left_panel_bottom.grid(row=10, column=0, sticky="s")
        self._gui.children["!frame"].rowconfigure(2, weight=1)
        left_panel_bottom.rowconfigure(0, weight=1)

        if sys.platform == "darwin":
            size = int(12 * self._default_os_scaling)
        else:
            size = int(12 * self._default_os_scaling * self._platform_scale)
        custom_function_button = tk.Button(
            master=left_panel_bottom,
            text="Custom\nFunction",
            width=len("Function") - self._platform_offset,
            font=("TkDefaultFont", size),
            bd=self._platform_border,
            command=self.dialog_box_new_function,
        )
        custom_function_button.grid(row=10, column=0, sticky="ews", padx=5, pady=10)
        self._custom_function_button = custom_function_button

    def hide_custom_function_button(self):
        if self._new_user_stage % 41 != 0:
            return
        self._custom_function_button.grid_forget()

    def show_custom_function_button(self):
        if self._new_user_stage % 41 != 0:
            self.create_custom_function_button()
            return

        self._custom_function_button.grid(row=10, column=0, sticky="ws", padx=5, pady=10)

    def dialog_box_new_function(self):

        dialog_box = tk.Toplevel()
        dialog_box.geometry(f"{round(self._os_width / 4)}x{round(self._os_height / 4)}")
        dialog_box.title("New Custom Function")
        if sys.platform == "win32":
            dialog_box.iconbitmap(f"{pkg_path()}/images/icon.ico")
        elif sys.platform == "darwin":
            photo = tk.PhotoImage(file=f"{pkg_path()}/images/splash.png")
            dialog_box.iconphoto(False, photo)
        elif sys.platform == "linux":
            icon = tk.PhotoImage(file=f"{pkg_path()}/images/icon.png")
            dialog_box.iconphoto(False, icon)
        else:
            pass

        data_frame = tk.Frame(master=dialog_box)
        data_frame.grid(row=0, column=0, sticky="ew")
        # data_frame.columnconfigure(1,minsize=500)
        exp_frame = tk.Frame(master=dialog_box)
        exp_frame.grid(row=1, column=0, sticky="w")

        name_label = tk.Label(master=data_frame, text="Function Name")
        name_label.grid(row=0, column=0, sticky="w")
        name_data = tk.Entry(master=data_frame, width=35)
        name_data.insert(0, "")
        name_data.grid(row=0, column=1, sticky="w")

        form_label = tk.Label(master=data_frame, text="Functional Form")
        form_label.grid(row=1, column=0, sticky="w")
        form_data = tk.Entry(master=data_frame, width=35)
        form_data.insert(0, "")
        form_data.grid(row=1, column=1, sticky="w")

        name_example_label = tk.Label(master=data_frame, text="\nName Example")
        name_example_label.grid(row=2, column=0, sticky="w")
        name_example_data = tk.Entry(master=data_frame, width=35)
        name_example_data.insert(0, "new_primitive")
        name_example_data.grid(row=2, column=1, sticky="ws")

        form_example_label = tk.Label(master=data_frame, text="Form Example")
        form_example_label.grid(row=3, column=0, sticky="w")
        form_example_data = tk.Entry(master=data_frame, width=35)
        form_example_data.insert(0, "np.atan(x)*scipy.special.jv(0,x)")
        form_example_data.grid(row=3, column=1, sticky="w")

        explanation_label1 = tk.Label(
            master=exp_frame, text="\nSupports numpy and scipy functions."
        )
        explanation_label1.grid(row=4, column=0, sticky="w")
        explanation_label2 = tk.Label(master=exp_frame, text="Avoid special characters and spaces.")
        explanation_label2.grid(row=5, column=0, sticky="w")
        explanation_label3 = tk.Label(
            master=exp_frame,
            text="The first letter of the name" " should come before 's' in the alphabet.",
        )
        explanation_label3.grid(row=6, column=0, sticky="w")

        close_dialog_button = tk.Button(
            master=data_frame,
            text="OK",
            width=len("OK") - self._platform_offset,
            # font=('TkDefaultFont', int(12*self._default_os_scaling*self._platform_scale)),
            bd=self._platform_border,
            command=self.close_dialog_box_command_custom_function,
        )
        close_dialog_button.grid(row=0, column=10, padx=5, pady=0, sticky="ns")
        dialog_box.bind("<Return>", self.close_dialog_box_command_custom_function)
        dialog_box.focus_force()

        self._popup_window = dialog_box
        self._gui.wait_window(dialog_box)

    # noinspection PyUnusedLocal
    def close_dialog_box_command_custom_function(
        self,
        bind_command=None,  # pylint: disable=unused-argument
    ):

        if self._popup_window is None:
            logger("Window already closed")
        name_str = self._popup_window.children["!frame"].children["!entry"].get()
        form_str = self._popup_window.children["!frame"].children["!entry2"].get()

        if " " in name_str:
            self.add_message("\nYou can't have a name with a space in it.")
            return
        if name_str == "":
            self.add_message("\nYou can't have a blank name.")
            return
        if name_str in [x for x in regex.split(" ", self._custom_function_names) if x]:
            self.add_message("\nYou can't reuse names.")
            return
        if " " in form_str:
            self.add_message("\nYou can't have a functional form with a space in it.")
            return
        if form_str == "":
            self.add_message("\nYou can't have a blank functional form.\n")
            return

        self._custom_function_names += f" {name_str}"
        self._custom_function_forms += f" {form_str}"
        self._changed_optimizer_opts_flag = True
        self.update_custom_checkbox()
        try:
            self.update_optimizer()
        except AttributeError as ae:
            self.add_message(f"\n \n> Custom Function Error: {str(ae)}")
            self.remove_named_custom(name_str)
            self.update_custom_checkbox()
            return
        self.save_defaults()
        self._popup_window.destroy()

        logger(f"Frontend.close_dialog(): custom names: >{self._custom_function_names}<")
        logger(f"Frontend.close_dialog(): custom defns: >{self._custom_function_forms}<")

    # RIGHT PANEL FUNCTIONS ----------------------------------------------------------------------->

    def add_message(self, message_string) -> bool:

        # TODO: consider also printing to a log file
        # logger("Add_message: ",self._gui.winfo_height(), message_string)
        text_frame = self._right_panel_frame  # self._gui.children['!frame3']
        # text_frame.update()  # WHY was this necessary? It hangs the mac on restart
        # logger("Add_message: ",self._gui.winfo_height(), message_string)

        for line in regex.split("\n", message_string):
            if line == "":
                continue
            my_font = "consolas", int(12 * self._default_os_scaling)
            if sys.platform == "darwin":
                my_font = "courier new bold", int(12 * self._default_os_scaling)
            new_message_label = tk.Label(
                master=text_frame,
                text=line,
                bg=self._console_color,
                fg=hexx(self._printout_color),
                font=my_font,
            )

            new_message_label.grid(row=self._num_messages_ever, column=0, sticky=tk.W)
            self._num_messages += 1
            self._num_messages_ever += 1

            new_message_label.update()  # required to scroll the console up when the buffer fills
            self._max_message_length = self._gui.winfo_height() // new_message_label.winfo_height()

            self._progress_label = new_message_label

        if self._num_messages > self._max_message_length:
            self.remove_n_messages(self._num_messages - self._max_message_length)
        return True

    def remove_n_messages(self, n):

        text_frame = self._right_panel_frame

        key_removal_list = []
        for key in text_frame.children:
            key_removal_list.append(key)
            if len(key_removal_list) == n:
                break

        for key in key_removal_list:
            text_frame.children[key].destroy()

        self._num_messages = len(text_frame.children)

    def print_results_to_console(self):
        print_string = ""
        if self.current_model.name == "Linear":
            if self.data_handler.logy_flag:
                print_string += "\n>  Linear fit is LY ="
            else:
                print_string += "\n>  Linear fit is y ="
            if self.data_handler.logx_flag:
                print_string += " m LX + b with\n"
            else:
                print_string += " m x + b with\n"
            args, uncs = self.current_args, self.current_uncs
            m, sigmam = args[0], uncs[0]
            b, sigmab = args[1], uncs[1]
            # the uncertainty for linear regression is also very well-studied, so
            # this should be a test case for uncertainty values
            print_string += f"   m = {m:+.2E}  \u00B1  {sigmam:.2E}\n"
            print_string += f"   b = {b:+.2E}  \u00B1  {sigmab:.2E}\n"
            # TODO: this needs to do something more complicated when fitting all
            print_string += (
                f"  Goodness of fit: R\U000000B2 = "
                f"{self.optimizer.r_squared(self.current_model):.4F}"
            )
            if self.criterion != "rchisqr":
                print_string += (
                    f"  ,  {self.criterion} = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
            else:
                print_string += (
                    f"  ,  {self.sym_chi}{sup(2)}/dof = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
        elif self.current_model.name[:10] == "Polynomial":
            deg = self._polynomial_degree_tkint.get()
            args = self.current_args
            uncs = (
                self.current_uncs if deg < self.max_poly_degree() else [0 for _ in range(deg + 1)]
            )
            if self.data_handler.logy_flag:
                print_string += "\n>  Polynomial fit is LY = "
            else:
                print_string += "\n>  Polynomial fit is y = "
            if self.data_handler.logx_flag:
                for n in range(deg):
                    print_string += f"C{sub(deg - n)}LX{sup(deg - n)}+"
            else:
                for n in range(deg):
                    print_string += f"C{sub(deg - n)}x{sup(deg - n)}+"
            print_string += f"C{sub(0)}\n   where the constants are"
            for n in range(deg + 1):
                val, sig = args[n], uncs[n]
                print_string += f"\n   C{sub(deg - n)} = {val:+.2E}  \u00B1  {sig:+.2E}\n"
            if self.criterion != "rchisqr":
                print_string += (
                    f"  Goodness of fit: {self.criterion} = " f"{self.current_rchisqr:.2F}\n"
                )
            else:
                print_string += (
                    f"  Goodness of fit: {self.sym_chi}{sup(2)}/dof = "
                    f"{self.current_rchisqr:.2F}\n"
                )
        elif self.current_model.name == "Normal":
            args, uncs = self.current_args, self.current_uncs
            if self.data_handler.logy_flag:
                print_string += f"\n>  {self.current_model.name} fit is LY ="
            else:
                print_string += f"\n>  {self.current_model.name} fit is y ="
            if self.data_handler.logx_flag:
                print_string += (
                    " 1/\u221A(2\u03C0\u03C3\U000000B2) "
                    "exp[-(LX-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
                )
            else:
                print_string += (
                    " 1/\u221A(2\u03C0\u03C3\U000000B2) "
                    "exp[-(x-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
                )
            sigma, sigmasigma = args[0], uncs[0]
            mu, sigmamu = args[1], uncs[1]
            print_string += f"   \u03BC = {mu:+.2E}  \u00B1  {sigmamu:.2E}\n"
            print_string += f"   \u03C3 =  {sigma:.2E}  \u00B1  {sigmasigma:.2E}\n"
            if self.criterion != "rchisqr":
                print_string += (
                    f"  Goodness of fit: {self.criterion} = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
            else:
                print_string += (
                    f"  Goodness of fit: {self.sym_chi}{sup(2)}/dof = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
            logger([datum.val for datum in self.data_handler.data])
        elif self.current_model.name == "Gaussian":
            args, uncs = self.current_args, self.current_uncs
            if self.data_handler.logy_flag:
                print_string += f"\n>  {self.current_model.name} fit is LY ="
            else:
                print_string += f"\n>  {self.current_model.name} fit is y ="
            if self.data_handler.logx_flag:
                print_string += " A exp[-(LX-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
            else:
                print_string += " A exp[-(x-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
            A, sigma_A = args[0], uncs[0]
            sigma, sigmasigma = args[1], uncs[1]
            mu, sigmamu = args[2], uncs[2]

            print_string += f"   A = {A:+.2E}  \u00B1  {sigma_A:.2E}\n"
            print_string += f"   \u03BC = {mu:+.2E}  \u00B1  {sigmamu:.2E}\n"
            print_string += f"   \u03C3 =  {sigma:.2E}  \u00B1  {sigmasigma:.2E}\n"
            if self.criterion != "rchisqr":
                print_string += (
                    f"  Goodness of fit: {self.criterion} = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
            else:
                print_string += (
                    f"  Goodness of fit: {self.sym_chi}{sup(2)}/dof = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
        elif self.current_model.name[-8:] == "Gaussian" and self._gaussian_modal_tkint.get() > 1:
            if self.data_handler.logy_flag:
                print_string += f"\n>  {self.current_model.name} fit is LY ="
            else:
                print_string += f"\n>  {self.current_model.name} fit is y ="
            for idx, _ in enumerate(self.current_model.children_list):
                if idx > 0:
                    print_string += "+"
                args, uncs = (
                    self.current_args[3 * idx : 3 * idx + 3],
                    self.current_uncs[3 * idx : 3 * idx + 3],
                )
                if self.data_handler.logx_flag:
                    print_string += (
                        f" A{sub(idx + 1)} exp[-(LX-\u03BC{sub(idx + 1)})\U000000B2/2"
                        f"\u03C3{sub(idx + 1)}\U000000B2] with\n"
                    )
                else:
                    print_string += (
                        f" A{sub(idx + 1)} exp[-(x-\u03BC{sub(idx + 1)})\U000000B2/2"
                        f"\u03C3{sub(idx + 1)}\U000000B2] with\n"
                    )
                A, sigma_A = args[0], uncs[0]
                sigma, sigmasigma = args[1], uncs[1]
                mu, sigmamu = args[2], uncs[2]

                print_string += f"   A{sub(idx + 1)} = {A:+.2E}  \u00B1  {sigma_A:.2E}\n"
                print_string += f"   \u03BC{sub(idx + 1)} = {mu:+.2E}  \u00B1  {sigmamu:.2E}\n"
                print_string += (
                    f"   \u03C3{sub(idx + 1)} =  " f"{sigma:.2E}  \u00B1  {sigmasigma:.2E}\n"
                )
            if self.criterion != "rchisqr":
                print_string += (
                    f"  Goodness of fit: {self.criterion} = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
            else:
                print_string += (
                    f"  Goodness of fit: {self.sym_chi}{sup(2)}/dof = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
        elif self.current_model.name == "Sigmoid":
            args, uncs = self.current_args, self.current_uncs
            if self.data_handler.logy_flag:
                print_string += "\n>  Sigmoid fit is LY ="
            else:
                print_string += f"\n>  {self.current_model.name} fit is y ="
            if self.data_handler.logx_flag:
                print_string += " F + H/(1 + exp[-(LX-x0)/w] ) with\n"
            else:
                print_string += " F + H/(1 + exp[-(x-x0)/w] ) with\n"
            F, sigma_F = args[0], uncs[0]
            H, sigma_H = args[1], uncs[1]
            w, sigma_w = args[2], uncs[2]
            x0, sigma_x0 = args[3], uncs[3]

            print_string += f"   F  = {F:+.2E}  \u00B1  {sigma_F:.2E}\n"
            print_string += f"   H  = {H:+.2E}  \u00B1  {sigma_H:.2E}\n"
            print_string += f"   w  =  {w:.2E}  \u00B1  {sigma_w:.2E}\n"
            print_string += f"   x0 = {x0:+.2E}  \u00B1  {sigma_x0:.2E}\n"
            if self.criterion != "rchisqr":
                print_string += (
                    f"  Goodness of fit: {self.criterion} = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
            else:
                print_string += (
                    f"  Goodness of fit: {self.sym_chi}{sup(2)}/dof = "
                    f"{self.optimizer.criterion(self.current_model):.2F}\n"
                )
        elif self._model_name_tkstr.get() == "Procedural":
            if self.data_handler.logy_flag:
                print_string += f"\n> Selected model is LY = {self.current_model.name}"
            else:
                print_string += f"\n> Selected model is y = {self.current_model.name}"
            if self.data_handler.logx_flag:
                print_string += f"(LX) w/ {self.current_model.dof} dof and where\n"
            else:
                print_string += f"(x) w/ {self.current_model.dof} dof and where\n"
            for idx, (par, unc) in enumerate(zip(self.current_args, self.current_uncs)):
                print_string += f"  c{idx} =  {par:+.2E}  \u00B1  {unc:.2E}\n"
            if self.criterion != "rchisqr":
                print_string += f"\n \n  This has {self.criterion} = "
            else:
                print_string += f"\n \n  This has {self.sym_chi}{sup(2)}/dof = "
            print_string += (
                f"{self.optimizer.criterion(self.current_model):.2F},"
                if self.optimizer.criterion(self.current_model) > 0.01
                else f"{self.optimizer.criterion(self.current_model):.2E},"
            )
            print_string += " and as a tree, this is \n"
            print_string += self.current_model.tree_as_string_with_args() + "\n"
        elif self._model_name_tkstr.get() == "Brute-Force":
            if self.data_handler.logy_flag:
                print_string += f"\n> Model is LY = {self.current_model.name}"
            else:
                print_string += f"\n> Model is y = {self.current_model.name}"
            if self.data_handler.logx_flag:
                print_string += f"(LX) w/ {self.current_model.dof} dof and where\n"
            else:
                print_string += f"(x) w/ {self.current_model.dof} dof and where\n"
            for idx, (par, unc) in enumerate(zip(self.current_args, self.current_uncs)):
                print_string += f"  c{idx} =  {par:+.2E}  \u00B1  {unc:.2E}\n"
            print_string += "\n \n> As a tree, this is \n"
            print_string += self.current_model.tree_as_string_with_args() + "\n"
        elif self._model_name_tkstr.get() == "Manual":
            if self.data_handler.logy_flag:
                print_string += f"\n> Selected model is LY = {self.current_model.name}"
            else:
                print_string += f"\n> Selected model is y = {self.current_model.name}"
            if self.data_handler.logx_flag:
                print_string += f"(LX) w/ {self.current_model.dof} dof and where\n"
            else:
                print_string += f"(x) w/ {self.current_model.dof} dof and where\n"
            for idx, (par, unc) in enumerate(zip(self.current_args, self.current_uncs)):
                print_string += f"  c{idx} =  {par:+.2E}  \u00B1  {unc:.2E}\n"
            if self.criterion != "rchisqr":
                print_string += f"\n \n> This has {self.criterion} = "
            else:
                print_string += f"\n \n> This has {self.sym_chi}{sup(2)}/dof = "
            print_string += (
                f"{self.optimizer.criterion(self.current_model):.2F},"
                if self.optimizer.criterion(self.current_model) > 0.01
                else f"{self.optimizer.criterion(self.current_model):.2E},"
            )
            print_string += " and as a tree, this is \n"
            print_string += self.current_model.tree_as_string_with_args() + "\n"
        else:
            logger(f"{self.current_model.name=} {self.data_handler.normalized=}")
            raise EnvironmentError
        if self.data_handler.logy_flag and self.data_handler.logx_flag:
            print_string += (
                f"Keep in mind that LY = log(y/{self.data_handler.Y0:.2E}) "
                f"and LX = log(x/{self.data_handler.X0:.2E})\n"
            )
        elif self.data_handler.logy_flag:
            print_string += f"Keep in mind that LY = log(y/{self.data_handler.Y0:.2E})\n"
        elif self.data_handler.logx_flag:
            print_string += f"Keep in mind that LX = log(x/{self.data_handler.X0:.2E})\n"
        self.add_message(print_string)

    def create_colors_console_menu(self):

        if self._new_user_stage % 67 == 0:
            return
        self._new_user_stage *= 67

        head_menu = tk.Menu(master=self._gui, tearoff=0)

        # This works on PC but not OSX
        # head_menu.add_cascade(label="Background Colour", menu=self._printout_background_menu)
        # head_menu.add_cascade(label="Message Colour", menu=self._printout_menu)

        background_menu = tk.Menu(master=head_menu, tearoff=0)
        background_menu.add_command(label="Default", command=self.console_color_default)
        background_menu.add_command(label="White", command=self.console_color_white)
        background_menu.add_command(label="Pale", command=self.console_color_pale)

        printout_menu = tk.Menu(master=head_menu, tearoff=0)
        printout_menu.add_command(label="Default", command=self.printout_color_default)
        printout_menu.add_command(label="White", command=self.printout_color_white)
        printout_menu.add_command(label="Black", command=self.printout_color_black)

        head_menu.add_cascade(label="Background Colour", menu=background_menu)
        head_menu.add_cascade(label="Message Colour", menu=printout_menu)

        self._colors_console_menu = head_menu

    def do_colors_console_popup(self, event: tk.Event):
        self.create_colors_console_menu()
        image_colors_menu: tk.Menu = self._colors_console_menu
        try:
            image_colors_menu.tk_popup(event.x_root, event.y_root)
        finally:
            image_colors_menu.grab_release()

    # MIDDLE PANEL FUNCTIONS ---------------------------------------------------------------------->

    def create_image_frame(self):  # !frame : image only
        image_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        image_frame.grid(row=0, column=0, sticky="w")
        self.load_splash_image()

    def create_data_perusal_frame(self):  # !frame2 : inspect, left<>right buttons
        self._data_perusal_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._data_perusal_frame.grid(row=1, column=0, sticky="ew")
        self._data_perusal_frame.grid_columnconfigure(0, weight=1)

        data_perusal_frame_left = tk.Frame(master=self._data_perusal_frame, bg=self._sbf)
        data_perusal_frame_left.grid(row=0, column=0, sticky="w")

        data_perusal_frame_right = tk.Frame(master=self._data_perusal_frame, bg=self._sbf)
        data_perusal_frame_right.grid(row=0, column=1, sticky="e")

    def create_fit_options_frame(
        self,
    ):  # !frame3 : fit type, procedural top5, pause/go, refit
        self._fit_options_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._fit_options_frame.grid(
            row=3, column=0, sticky="w"
        )  # row2 is reserved for the black line

    def create_plot_options_frame(self):  # !frame4 : logx, logy, normalize
        self._gui.children["!frame2"].columnconfigure(1, minsize=50)
        self._plot_options_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._plot_options_frame.grid(row=0, column=1, sticky="ns")

    # def create_linear_frame(self) : pass
    def create_polynomial_frame(self):  # !frame6 : depth of procedural fits
        self._polynomial_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._polynomial_frame.grid(row=4, column=0, sticky="w")

    def create_gaussian_frame(self):  # !frame7 : depth of procedural fits
        self._gaussian_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._gaussian_frame.grid(row=4, column=0, sticky="w")

    # def create_sigmoid_frame(self) : pass
    def create_procedural_frame(self):  # !frame5 : checkboxes, depth of procedural fit
        self._procedural_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._procedural_frame.grid(row=4, column=0, sticky="w")

    # def create_brute_force_frame(self) : pass
    def create_manual_frame(self):  # !frame6 : depth of procedural fits
        self._manual_frame = tk.Frame(master=self._gui.children["!frame2"], bg=self._sbf)
        self._manual_frame.grid(row=4, column=0, sticky="w")

    # IMAGE frame --------------------------------------------------------------------------------->
    def load_splash_image(self):
        self._image_path = f"{pkg_path()}/images/splash.png"

        img_raw = Image.open(self._image_path)
        if self._default_gui_width < 2 and self._image_r == 1:
            self._image_r = (8 / 9) * self._os_height / (2 * img_raw.height)
        img_resized = img_raw.resize(
            (
                round(img_raw.width * self._image_r),
                round(img_raw.height * self._image_r),
            )
        )
        self._image_path = f"{self._image_path[:-4]}_mod.png"
        img_resized.save(fp=self._image_path)

        self._image = tk.PhotoImage(file=self._image_path)
        self._image_frame = tk.Label(
            master=self._gui.children["!frame2"].children["!frame"],
            image=self._image,
            relief=tk.SUNKEN,
            bg=self._sbf,
        )
        logger(f"Created frame {self._image_frame}")
        self._image_frame.grid(row=0, column=0)
        self._image_frame.grid_propagate(True)
        self._image_frame.bind(self._right_click, self.do_colors_image_popup)
        self._image_frame.bind("<MouseWheel>", self.do_image_resize)
        if sys.platform == "linux":
            self._image_frame.bind("<Button-4>", self.mouse_wheel_up)
            self._image_frame.bind("<Button-5>", self.mouse_wheel_down)

    def switch_image(self):
        self._image = tk.PhotoImage(file=self._image_path)
        self._image_frame.configure(image=self._image)

    # noinspection PyUnusedLocal
    def mouse_wheel_up(self, event):  # pylint: disable=unused-argument
        up = tk.Event()
        up.delta = +120
        self.do_image_resize(event=up)

    # noinspection PyUnusedLocal
    def mouse_wheel_down(self, event):  # pylint: disable=unused-argument
        down = tk.Event()
        down.delta = -120
        self.do_image_resize(event=down)

    def do_image_resize(self, event):

        # logger(type(event))
        d = event.delta / 120
        self._image_r *= 1 + d / 10
        self.save_defaults()

        if self._image_path in [
            f"{pkg_path()}/images/splash.png",
            f"{pkg_path()}/images/splash_mod.png",
        ]:
            raw: Image = Image.open(f"{pkg_path()}/images/splash.png")
            resized = raw.resize(
                (round(raw.width * self._image_r), round(raw.height * self._image_r))
            )
            self._image_path = f"{pkg_path()}/images/splash_mod.png"
            resized.save(fp=self._image_path)
            self.switch_image()
            return

        if self._showing_fit_all_image:
            self.fit_all_command()
            # self.save_show_fit_all()  # this contains switch_image()
        elif self._showing_fit_image:
            self.save_show_fit_image()  # this contains switch_image()
        else:
            self.show_data()  # this contains switch_image()

    def create_colors_image_menu(self):

        if self._new_user_stage % 61 == 0:
            return
        self._new_user_stage *= 61

        head_menu = tk.Menu(master=self._gui, tearoff=0)

        # This commented part works on PC but nto OSX
        # head_menu.add_cascade(label="Background Colour", menu=self._background_menu)
        # head_menu.add_cascade(label="Data/Axis Colour", menu=self._dataaxis_menu)
        # head_menu.add_cascade(label="Fit Colour", menu=self._fit_colour_menu)

        background_menu = tk.Menu(master=head_menu, tearoff=0)
        background_menu.add_command(label="Default", command=self.bg_color_default)
        background_menu.add_command(label="White", command=self.bg_color_white)
        background_menu.add_command(label="Dark", command=self.bg_color_dark)
        background_menu.add_command(label="Black", command=self.bg_color_black)

        dataaxis_menu = tk.Menu(master=head_menu, tearoff=0)
        dataaxis_menu.add_command(label="Default", command=self.dataaxes_color_default)
        dataaxis_menu.add_command(label="White", command=self.dataaxes_color_white)

        fit_colour_menu = tk.Menu(master=head_menu, tearoff=0)
        fit_colour_menu.add_command(label="Default", command=self.fit_color_default)
        fit_colour_menu.add_command(label="White", command=self.fit_color_white)
        fit_colour_menu.add_command(label="Black", command=self.fit_color_black)

        head_menu.add_cascade(label="Background Colour", menu=background_menu)
        head_menu.add_cascade(label="Data/Axis Colour", menu=dataaxis_menu)
        head_menu.add_cascade(label="Fit Colour", menu=fit_colour_menu)

        self._colors_image_menu = head_menu

    def do_colors_image_popup(self, event: tk.Event):
        self.create_colors_image_menu()
        image_colors_menu: tk.Menu = self._colors_image_menu
        try:
            image_colors_menu.tk_popup(event.x_root, event.y_root)
        finally:
            image_colors_menu.grab_release()

    # DATA PERUSAL frame -------------------------------------------------------------------------->
    def create_inspect_button(self):

        # TODO: also make a save figure button

        # inspect_bar
        # self._gui.children['!frame2'].children['!frame2'].children['!frame']

        data_perusal_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame2"].children["!frame"],
            text="Inspect",
            width=len("Inspect") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.inspect_command,
        )
        data_perusal_button.grid(row=0, column=0, padx=5, pady=5)

    def inspect_command(self):

        plt.show()

        # TODO: find a way to show() again without rerunning fits

        if self._showing_fit_all_image:
            self.fit_all_command(quiet=True)
        elif self._showing_fit_image:
            self.save_show_fit_image()
        else:
            self.show_data()

    def show_left_right_buttons(self):

        if self._new_user_stage % 5 == 0:
            return
        self._new_user_stage *= 5

        left_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame2"].children["!frame"],
            text=self.sym_left,
            bd=self._platform_border,
            command=self.image_left_command,
        )
        count_text = tk.Label(
            master=self._gui.children["!frame2"].children["!frame2"].children["!frame"],
            text=f"{self._curr_image_num % len(self._data_handlers) + 1}"
            f"/{len(self._data_handlers)}",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bg=self._sbf,
        )
        right_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame2"].children["!frame"],
            text=self.sym_right,
            bd=self._platform_border,
            command=self.image_right_command,
            bg=self._sbf,
        )
        left_button.grid(row=0, column=1, padx=5, pady=5)
        count_text.grid(row=0, column=2)
        right_button.grid(row=0, column=3, padx=5, pady=5)

    def image_left_command(self):
        self._curr_image_num = (self._curr_image_num - 1) % len(self._data_handlers)
        self.image_change_command()

    def image_right_command(self):
        self._curr_image_num = (self._curr_image_num + 1) % len(self._data_handlers)
        self.image_change_command()

    def image_change_command(self):
        self._changed_data_flag = True
        self._showing_fit_all_image = False

        if self._showing_fit_image:
            if self._refit_on_click:
                self.show_current_data_with_fit()  # for refitting
            else:
                self.save_show_fit_image()
        else:
            self.show_current_data()

        self.update_data_select()

        if self.data_handler.histogram_flag:
            self.show_normalize_button()
        else:
            self.hide_normalize_button()
        self.update_logx_relief()
        self.update_logy_relief()

        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()

    def create_error_bands_button(self):
        self._error_bands_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame2"].children["!frame2"],
            text="Error Bands",
            width=len("Error Bands") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.show_error_bands_command,
        )
        self._error_bands_button.grid(row=0, column=0, pady=5, sticky="e")
        self._show_error_bands = 0

    def show_error_bands_command(self):
        if any(np.isinf(self.current_uncs)):
            self.add_message("\n \n> Can't produce error bands with infinite uncertainty.")
            self._error_bands_button.configure(text="Error Bands")
            self._show_error_bands = 0
            return
        if self._error_bands_button["text"] == "Error Bands":
            logger(f"Switching to 1-{self.sym_sigma} confidence region")
            self._error_bands_button.configure(text=f"       1-{self.sym_sigma}       ")
            self._show_error_bands = 1
        elif self._error_bands_button["text"] == f"       1-{self.sym_sigma}       ":
            logger(f"Switching to 2-{self.sym_sigma} confidence region")
            self._error_bands_button.configure(text=f"       2-{self.sym_sigma}       ")
            self._show_error_bands = 2
        elif self._error_bands_button["text"] == f"       2-{self.sym_sigma}       ":
            logger("Switching to both confidence regions")
            self._error_bands_button.configure(text="Both")
            self._show_error_bands = 3
        elif self._error_bands_button["text"] == "Both":
            logger("Switching off error bands")
            self._error_bands_button.configure(text="Error Bands")
            self._show_error_bands = 0
        else:
            logger("Can't change from", self._error_bands_button["text"])

        if self._showing_fit_all_image:
            self.fit_all_command(quiet=True)
            return
        self.save_show_fit_image()

    def create_residuals_button(self):
        self._residuals_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame2"].children["!frame2"],
            text="Residuals",
            width=len("Residuals") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.show_residuals_command,
        )
        self._residuals_button.grid(row=0, column=1, padx=5, pady=5, sticky="e")

    def show_residuals_command(self):
        # TODO: this should do something different when fitting all
        # this also doesn't make sense when using LX/LY
        #    -- the residuals plot looks good but I don't think the normality tests are working
        #    --   maybe it's because we aren't normalizing the residuals histogram?

        # TODO: also show the time residuals as a function of x,
        #  below plot, toggle with residuals button
        if self.current_model is None:
            logger("Residuals_command: you shouldn't be here, quitting")
            raise SystemExit

        logger(f"\n\n\n\n\n\n\nShowing residuals relative to {self.current_model.name}")

        res_filepath = f"{pkg_path()}/plots/residuals.csv"

        residuals = []
        norm_residuals = []
        with open(file=res_filepath, mode="w", encoding="utf-8") as res_file:
            if self._showing_fit_all_image:
                for handler in self._data_handlers:
                    for datum in handler.data:
                        res = datum.val - self.current_model.eval_at(datum.pos)
                        residuals.append(res)
                        norm_residuals.append(res / datum.sigma_val)
                        res_file.write(f"{res},\n")
            else:
                for datum in self.data_handler.data:
                    # logger(datum)
                    res = datum.val - self.current_model.eval_at(datum.pos)
                    residuals.append(res)
                    norm_residuals.append(res / datum.sigma_val)
                    res_file.write(f"{res},\n")

        res_handler = DataHandler(filepath=res_filepath)
        res_optimizer = Optimizer(data=res_handler.data)

        sample_mean = sum(residuals) / len(residuals)
        sample_variance = sum(((res - sample_mean) ** 2 for res in residuals)) / (
            len(residuals) - 1
        )
        sample_std_dev = np.sqrt(sample_variance)

        # should be counting with std from 0, not from residual_mean
        variance_rel_zero = sum((res**2 for res in residuals)) / (len(residuals) - 1)
        std_dev_rel_zero = np.sqrt(variance_rel_zero)

        std_dev = std_dev_rel_zero
        sample_mean = 0.0

        # actually, you should be using the mean and sigma according to the *fit*
        if len(res_handler.data) >= 4:  # shouldn't fit a gaussian to 3 points
            res_optimizer.fit_this_and_get_model_and_covariance(
                model_=CompositeFunction.built_in("Gaussian")
            )
            _, sigma, x0 = res_optimizer.shown_parameters
            _, sigmasigma, sigmax0 = res_optimizer.shown_uncertainties
            logger(f"Mean from fit: {x0} +- {sigmax0}")
            logger(
                f"Sigma from fit: {sigma} +- {sigmasigma} "
                f"... sample standard deviation: {sample_std_dev}"
            )
            std_dev = sigma
            sample_mean = x0
        else:
            np_residuals = np.array(residuals)
            mu = np_residuals.mean()
            sigma = np_residuals.std() if np_residuals.std() != 0 else 1
            count = len(residuals)
            manual_gaussian = CompositeFunction.built_in("Gaussian")
            manual_gaussian.set_args(
                count * res_handler.bin_width() / np.sqrt(2 * np.pi * sigma**2),
                sigma,
                mu,
            )
            logger(f"{res_handler.bin_width()=}")
            logger(
                count * res_handler.bin_width() / np.sqrt(2 * np.pi * sigma**2),
                sigma,
                mu,
            )
            res_optimizer.shown_model = manual_gaussian
        res_optimizer.show_fit()

        if max(residuals) ** 2 + min(residuals) ** 2 < 1e-20:
            self.add_message("\n \n> Normality tests are not shown for a perfect fit.")
            return

        # rule of thumb
        num_within_error_bars = sum((1 if -1 < norm_res < 1 else 0 for norm_res in norm_residuals))

        touching_min = 0
        while True:
            binomial_lowweight = scipy.stats.binom.cdf(touching_min, len(residuals), 0.682689)
            if binomial_lowweight > 0.05:
                break
            touching_min += 1

        touching_max = len(residuals)
        while True:
            binomial_highweight = 1 - scipy.stats.binom.cdf(touching_max, len(residuals), 0.682689)
            if binomial_highweight > 0.05:
                break
            touching_max -= 1

        if any(np.isinf(norm_residuals)):
            logger("Can't do rule of thumb!")
        else:
            self.add_message("\n ")
            self.add_message(
                f"> By the 68% rule of thumb, the number of datapoints with error bars \n"
                f"   touching the line of best fit should obey "
                f"{touching_min} ≤ {num_within_error_bars} ≤ {touching_max}"
            )
            if touching_min <= num_within_error_bars <= touching_max:
                self.add_message(
                    "   Since this is obeyed, a very rough check has been passed that \n"
                    "   the fit is a proper representation of the data."
                )
            elif touching_min > num_within_error_bars:
                self.add_message(
                    "   Since this undershoots the minimum expected number, "
                    "it is likely that either \n"
                    "     • the fit is a poor representation of the data\n"
                    "     • the error bars have been underestimated\n"
                    "     • you are fitting multiple datasets\n"
                )
            elif touching_max < num_within_error_bars:
                self.add_message(
                    "   Since this exceeds the maximum expected number, either the data has been\n"
                    "   generated with an exact function, or the "
                    "error bars have been overestimated.\n"
                    "   In either case, it is likely that a good "
                    "model of the dataset has been found!"
                )

        count_ulow = sum(
            (1 if res - sample_mean < sample_mean - 2 * std_dev else 0 for res in residuals)
        )
        count_low = sum(
            (
                (1 if sample_mean - 2 * std_dev < res - sample_mean < sample_mean - std_dev else 0)
                for res in residuals
            )
        )
        count_middle = sum(
            (
                (1 if sample_mean - std_dev < res - sample_mean < sample_mean + std_dev else 0)
                for res in residuals
            )
        )
        count_high = sum(
            (
                (1 if sample_mean + std_dev < res - sample_mean < sample_mean + 2 * std_dev else 0)
                for res in residuals
            )
        )
        count_uhigh = sum(
            (1 if res - sample_mean > sample_mean + 2 * std_dev else 0 for res in residuals)
        )

        # if we have independent trials, a binomial distribution for a sample of size N
        # will have 90% confidence regions between kmin and kmax
        kmin_fartail = 0
        while True:
            binomial_lowweight = scipy.stats.binom.cdf(kmin_fartail, len(residuals), 0.0227505)
            if binomial_lowweight > 0.05:
                break
            kmin_fartail += 1

        kmax_fartail = len(residuals)
        while True:
            binomial_highweight = 1 - scipy.stats.binom.cdf(kmax_fartail, len(residuals), 0.0227505)
            if binomial_highweight > 0.05:
                break
            kmax_fartail -= 1

        kmin_tail = 0
        while True:
            binomial_lowweight = scipy.stats.binom.cdf(kmin_tail, len(residuals), 0.135905)
            if binomial_lowweight > 0.05:
                break
            kmin_tail += 1

        kmax_tail = len(residuals)
        while True:
            binomial_highweight = 1 - scipy.stats.binom.cdf(kmax_tail, len(residuals), 0.135905)
            if binomial_highweight > 0.05:
                break
            kmax_tail -= 1

        kmin_centre = 0
        while True:
            binomial_lowweight = scipy.stats.binom.cdf(kmin_centre, len(residuals), 0.682689)
            if binomial_lowweight > 0.05:
                break
            kmin_centre += 1

        kmax_centre = len(residuals)
        while True:
            binomial_highweight = 1 - scipy.stats.binom.cdf(kmax_centre, len(residuals), 0.682689)
            if binomial_highweight > 0.05:
                break
            kmax_centre -= 1

        logger(
            f"If residuals were normally distributed, "
            f"{kmin_fartail} ≤ {count_ulow} ≤ {kmax_fartail} "
        )
        logger(
            f"If residuals were normally distributed, " f"{kmin_tail} ≤ {count_low} ≤ {kmax_tail} "
        )
        logger(
            f"If residuals were normally distributed, "
            f"{kmin_centre} ≤ {count_middle} ≤ {kmax_centre} "
        )
        logger(
            f"If residuals were normally distributed, " f"{kmin_tail} ≤ {count_high} ≤ {kmax_tail} "
        )
        logger(
            f"If residuals were normally distributed, "
            f"{kmin_fartail} ≤ {count_uhigh} ≤ {kmax_fartail} "
        )
        pvalue_ulow = 1 if kmin_fartail <= count_ulow <= kmax_fartail else 0.1
        pvalue_low = 1 if kmin_tail <= count_low <= kmax_tail else 0.1
        pvalue_middle = 1 if kmin_centre <= count_middle <= kmax_centre else 0.1
        pvalue_high = 1 if kmin_tail <= count_high <= kmax_tail else 0.1
        pvalue_uhigh = 1 if kmin_fartail <= count_uhigh <= kmax_fartail else 0.1
        logger([pvalue_ulow, pvalue_low, pvalue_middle, pvalue_high, pvalue_uhigh])

        if 0.1 in [pvalue_ulow, pvalue_low, pvalue_middle, pvalue_high, pvalue_uhigh]:
            self.add_message(
                "\n \n> Based on residuals binned in the tail, far-tail, and central regions,"
            )
            if (
                self.data_handler.logx_flag
                or self.data_handler.logy_flag
                or self._showing_fit_all_image
            ):
                self.add_message(
                    f"  the probability that the residuals are normally distributed is "
                    f"{pvalue_ulow * pvalue_low * pvalue_middle * pvalue_high * pvalue_uhigh:.5F}\n"
                )
            else:
                self.add_message(
                    f"  the probability that you have found the correct fit for the data is "
                    f"{pvalue_ulow * pvalue_low * pvalue_middle * pvalue_high * pvalue_uhigh:.5F}\n"
                )

        self.add_message("\n \n> p-values from standard normality tests:\n")
        # other normality tests
        W, alpha = scipy.stats.shapiro(residuals)  # free mean, free variance
        logger(f"\n{W=} {alpha=}")
        self.add_message(f"  Shapiro-Wilk       : p = {alpha:.5F}")

        A2, crit, sig = scipy.stats.anderson(residuals, dist="norm")  # free mean, free variance
        logger(f"{A2=} {crit=} {sig=}")
        threshold_idx = -1
        for idx, icrit in enumerate(crit):
            if A2 > icrit:
                threshold_idx = idx
        if threshold_idx < 0:
            self.add_message(f"  Anderson-Darling   : p > {sig[0] * 0.01:.2F}")
        else:
            self.add_message(f"  Anderson-Darling   : p < {sig[threshold_idx] * 0.01:.2F}")

        # kolmogorov, kol_pvalue = scipy.stats.kstest(residuals,'norm')
        kolmogorov, kol_pvalue = scipy.stats.kstest(
            self.sample_standardize(residuals), "norm"
        )  # mean 0, variance 1
        logger(f"{kolmogorov=} {kol_pvalue=}")
        if kol_pvalue > 1e-5:
            self.add_message(f"  Kolmogorov-Smirnov : p = {kol_pvalue:.5F}")
        else:
            self.add_message(f"  Kolmogorov-Smirnov : p = {kol_pvalue:.2E}")

        if len(residuals) > 8:
            dagostino, dag_pvalue = scipy.stats.normaltest(residuals)  # free mean, free variance
            logger(f"{dagostino=} {dag_pvalue=}")
            if dag_pvalue > 1e-5:
                self.add_message(f"  d'Agostino         : p = {dag_pvalue:.5F}")
            else:
                self.add_message(f"  d'Agostino         : p = {dag_pvalue:.2E}")

        # TODO: I've noticed that there's a bad interaction with all tests when logging-y

    @staticmethod
    def sample_standardize(sample):
        np_sample = np.array(sample)
        mu = np_sample.mean()
        sigma = np_sample.std()
        return list((np_sample - mu) / sigma)

    # FIT OPTIONS frame --------------------------------------------------------------------------->
    def show_function_dropdown(self):
        if self._new_user_stage % 7 == 0:
            return
        self._new_user_stage *= 7

        # black line above frame 3
        self._gui.children["!frame2"].rowconfigure(2, minsize=1)
        black_line_as_frame = tk.Frame(master=self._gui.children["!frame2"], bg="black")
        black_line_as_frame.grid(row=2, column=0, sticky="ew")

        func_list = [
            "Linear",
            "Polynomial",
            "Gaussian",
            "Sigmoid",
            "Procedural",
            "Brute-Force",
            "Manual",
        ]

        # self._model_name_tkstr = tk.StringVar(self._fit_options_frame)
        self._model_name_tkstr.set(self._default_fit_type)

        function_dropdown = tk.OptionMenu(
            self._fit_options_frame, self._model_name_tkstr, *func_list
        )
        function_dropdown.configure(width=9)
        function_dropdown.configure(
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            )
        )
        options = self._fit_options_frame.nametowidget(function_dropdown.menuname)
        options.configure(
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            )
        )
        function_dropdown.grid(row=0, column=0)

        self._model_name_tkstr.trace("w", self.function_dropdown_trace)

    # noinspection PyUnusedLocal
    def function_dropdown_trace(self, *args):  # pylint: disable=unused-argument

        model_choice = self._model_name_tkstr.get()

        if model_choice == "Polynomial":
            self.show_degree_buttons()
        else:
            self.hide_degree_buttons()

        if model_choice == "Gaussian":
            self.show_modal_buttons()
        else:
            self.hide_modal_buttons()

        if model_choice != "Brute-Force":
            self.load_defaults()
            self.brute_forcing = False
            self.hide_pause_button()

        if model_choice in ["Procedural", "Brute-Force"]:
            if self._optimizer and len(self._optimizer.top5_args) > 0:
                self.show_top5_dropdown()
                self.update_top5_dropdown()
        else:
            self.hide_top5_dropdown()
            if model_choice == "Manual" and self._manual_model is None:
                pass
            else:
                self.fit_data_command()
        if model_choice == "Procedural":
            self.show_procedural_options()
        else:
            self.hide_procedural_options()

        if model_choice == "Manual":
            self.show_manual_fields()
            self._use_func_dict_name_tkbool["custom"].set(value=True)
            self._changed_optimizer_opts_flag = True
            self.update_optimizer()
        else:
            self.hide_manual_fields()

        if model_choice in ["Procedural", "Brute-Force", "Manual"]:
            self.show_custom_function_button()
        else:
            self.hide_custom_function_button()

    def create_top5_dropdown(self):

        if self._new_user_stage % 29 == 0:
            return
        self._new_user_stage *= 29

        top5_list = [
            f"{rx_sqr:.2F}: {name}"
            for rx_sqr, name in zip(self._optimizer.top5_rchisqrs, self._optimizer.top5_names)
        ]

        max_len = max((len(x) for x in top5_list))

        self._which5_name_tkstr = tk.StringVar(self._fit_options_frame)
        self._which5_name_tkstr.set("Top 5")

        top5_dropdown = tk.OptionMenu(self._fit_options_frame, self._which5_name_tkstr, *top5_list)
        top5_dropdown.configure(width=max_len - self._platform_offset)

        top5_dropdown.configure(
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            )
        )
        options = self._fit_options_frame.nametowidget(top5_dropdown.menuname)
        options.configure(
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            )
        )
        top5_dropdown.grid(row=0, column=1)

        self._which_tr_id = self._which5_name_tkstr.trace_add("write", self.which5_dropdown_trace)
        # ^ trace_add used to be trace which is deprecated

    def set_which5_no_trace(self, arg):
        self._which5_name_tkstr.trace_vdelete("w", self._which_tr_id)
        self._which5_name_tkstr.set(arg)
        self._which_tr_id = self._which5_name_tkstr.trace("w", self.which5_dropdown_trace)

    # noinspection PyUnusedLocal
    def which5_dropdown_trace(self, *args) -> None:  # pylint: disable=unused-argument
        which5_choice = self._which5_name_tkstr.get()
        logger(f"Changed top5_dropdown to {which5_choice}")
        # show the fit of the selected model
        _, model_name = regex.split(" ", which5_choice)
        try:
            selected_model_idx = self.optimizer.top5_names.index(model_name)
        except ValueError:
            logger(f"{model_name=} is not in {self.optimizer.top5_names}")
            selected_model_idx = 0

        self.current_model = self.optimizer.top5_models[selected_model_idx]
        self.current_covariance = self.optimizer.top5_covariances[selected_model_idx]
        self.current_rchisqr = self.optimizer.top5_rchisqrs[selected_model_idx]

        # also update the fit of the current model
        logger(f"{self._refit_on_click=} {self._changed_data_flag=}")
        if self._refit_on_click:
            # and self._changed_data_flag:
            logger("||| REFIT ON CLICK |||")
            self.show_current_data_with_fit(do_halving=True)
            return

        self.save_show_fit_image()
        self.print_results_to_console()

    def update_top5_dropdown(self):
        # uses the top5 models and rchiqrs in self.optimizer to populate the list
        if self._new_user_stage % 29 != 0:
            return

        # update options
        top5_dropdown: tk.OptionMenu = self._fit_options_frame.children["!optionmenu2"]

        curr_max = top5_dropdown["width"]
        top5_dropdown["menu"].delete(0, tk.END)
        top5_list = [
            f"{rx_sqr:.2F}: {name}" if rx_sqr < 1000 else f"----: {name}"
            for rx_sqr, name in zip(self.optimizer.top5_rchisqrs, self.optimizer.top5_names)
        ]
        for label in top5_list:
            top5_dropdown["menu"].add_command(
                # pylint: disable=protected-access
                label=label,
                command=tk._setit(self._which5_name_tkstr, label),
            )
        new_max = max((len(x) for x in top5_list))
        if new_max > int(curr_max):
            top5_dropdown.configure(width=new_max)
        # update label
        # selected_model_idx = self.optimizer.top5_names.index(self.current_model.name)
        # chisqr = self.optimizer.top5_rchisqrs[selected_model_idx]
        # name = self.optimizer.top5_names[selected_model_idx]
        # self.set_which5_no_trace(f"{chisqr:.2F}: {name}")
        self.set_which5_no_trace(
            f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
        )

    def update_top5_chisqrs(self):
        # uses the current top5 models to find rchisqrs when overlaid on new data
        # this is a very slow routine for a background process
        if self._new_user_stage % 29 != 0:
            return
        if self._refit_on_click:
            for idx, model in enumerate(self.optimizer.top5_models[:]):
                better_fit, _ = self.optimizer.fit_this_and_get_model_and_covariance(
                    model_=model, change_shown=False, do_halving=True
                )
                self.optimizer.top5_rchisqrs[idx] = self.optimizer.criterion(better_fit)
                logger(f"Update top 5 {model} {self.optimizer.top5_rchisqrs[idx]}")
        else:
            self.optimizer.update_top5_rchisqrs_for_new_data(self.data_handler.data)
        self.update_top5_dropdown()

    def hide_top5_dropdown(self):
        if self._new_user_stage % 29 != 0:
            return
        top5_dropdown = self._fit_options_frame.children["!optionmenu2"]
        top5_dropdown.grid_forget()
        self.hide_refit_button()

    def show_top5_dropdown(self):
        if self._new_user_stage % 29 != 0:
            self.create_top5_dropdown()
            return
        top5_dropdown = self._fit_options_frame.children["!optionmenu2"]
        top5_dropdown.grid(row=0, column=1)
        if len(self._data_handlers) > 1:
            logger("MAKING REFIT BUTTON")
            self.show_refit_button()

    def create_refit_button(self):

        if self._new_user_stage % 43 == 0 or self._new_user_stage % 29 != 0:
            return
        self._new_user_stage *= 43

        self._refit_button = tk.Button(
            self._fit_options_frame,
            text="Refit",
            width=len("Refit") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.refit_command,
        )
        self._refit_button.grid(row=0, column=3, padx=5, sticky="nw")

    def refit_command(self):
        self.show_current_data_with_fit(do_halving=True)
        self.set_which5_no_trace(
            f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
        )

    def hide_refit_button(self):
        if self._new_user_stage % 43 != 0:
            return
        self._refit_button.grid_forget()

    def show_refit_button(self):
        if self._refit_on_click:
            return
        if self._new_user_stage % 43 != 0:
            self.create_refit_button()
            return
        self._refit_button.grid(row=0, column=3, padx=5, sticky="nw")

    # PLOT OPTIONS frame -------------------------------------------------------------------------->
    def show_log_buttons(self):
        if self._new_user_stage % 13 == 0:
            return
        self._new_user_stage *= 13
        self._logx_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame4"],
            text="Log X",
            width=len("Log X") - self._platform_offset + 1,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.logx_command,
        )
        self._logx_button.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        self._logy_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame4"],
            text="Log Y",
            width=len("Log Y") - self._platform_offset + 1,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.logy_command,
        )
        self._logy_button.grid(row=1, column=0, padx=5, sticky="w")

    def logx_command(self):
        # TODO: if logging x or y, the top5 functions should be reset
        # flip-flop
        if self.data_handler.logx_flag:
            self.data_handler.logx_flag = False
        else:
            self.data_handler.logx_flag = True
            if not self.data_handler.logx_flag:
                self.add_message(
                    "\n \n> You can't log the x-data if there are non-positive numbers!"
                )

        self.update_logx_relief()

        self._changed_data_flag = True
        self.update_optimizer()

        if self._showing_fit_image:
            self.show_current_data_with_fit()
        else:
            self.show_current_data()

    def logy_command(self):
        # TODO: clear the top5 models list, since the top5 stored models fit "different" data
        if self.data_handler.logy_flag:
            self.data_handler.logy_flag = False
        else:
            self.data_handler.logy_flag = True
            if not self.data_handler.logy_flag:
                self.add_message(
                    "\n \n> You can't log the y-data if there are non-positive numbers!"
                )

        self.update_logy_relief()

        self._changed_data_flag = True
        self.update_optimizer()

        if self._showing_fit_image:
            self.show_current_data_with_fit()
        else:
            self.show_current_data()

        # TODO: clear the top5 models list, since the top5 stored models fit "different" data

    def update_logx_relief(self):
        if self._new_user_stage % 13 != 0:
            return
        if self.data_handler.logx_flag:
            # logger("Making log_x sunken")
            self._logx_button.configure(relief=tk.SUNKEN)
            self._logx_button.configure(bg="grey90")
            self.hide_normalize_button()
            logger(self._logx_button["relief"])
            return
        # logger("Making log_x raised")
        self._logx_button.configure(relief=tk.RAISED)

        if not self.data_handler.logy_flag and self.data_handler.histogram_flag:
            self.show_normalize_button()

    def update_logy_relief(self):
        if self._new_user_stage % 13 != 0:
            return

        if self.data_handler.logy_flag:
            # logger("Making log_y sunken")
            self._logy_button.configure(relief=tk.SUNKEN)
            self._logy_button.configure(bg="grey90")
            self.hide_normalize_button()
            return
        # logger("Making log_y raised")
        self._logy_button.configure(relief=tk.RAISED)

        if not self.data_handler.logx_flag and self.data_handler.histogram_flag:  # purpose?
            self.show_normalize_button()

    def create_normalize_button(self):
        if self._new_user_stage % 17 == 0:
            return
        self._new_user_stage *= 17

        self._normalize_button = tk.Button(
            master=self._gui.children["!frame2"].children["!frame4"],
            text="Normalize",
            width=len("Normalize") - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.normalize_command,
        )
        self._normalize_button.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    def normalize_command(self):
        if self.data_handler.normalized:
            self.add_message(
                "\n \nCan't de-normalize a histogram. You'll have to restart AutoFit.\n"
            )
            return
        status_good = self.data_handler.normalize_histogram_data()
        if not status_good:
            return
        self._normalize_button.configure(relief=tk.SUNKEN)
        self._normalize_button.configure(bg="grey90")
        if self._showing_fit_image:
            self.show_current_data_with_fit()
        else:
            self.show_current_data()
        self._changed_data_flag = True

    def hide_normalize_button(self):
        if self._new_user_stage % 17 != 0:
            return
        self._normalize_button.grid_forget()

    def show_normalize_button(self):
        if self._new_user_stage % 17 != 0:
            self.create_normalize_button()
            return
        self._normalize_button.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        if self.data_handler.normalized:
            self._normalize_button.configure(relief=tk.SUNKEN)
            self._normalize_button.configure(bg="grey90")
        else:
            self._normalize_button.configure(relief=tk.RAISED)
            self._normalize_button.configure(bg=self._sbf)

    def update_data_select(self):
        # left frame
        text_label: tk.Label = self._data_perusal_frame.children["!frame"].children["!label"]
        if self._showing_fit_all_image:
            text_label.configure(
                text=f"-/{len(self._data_handlers)}",
                font=(
                    "TkDefaultFont",
                    int(12 * self._default_os_scaling * self._platform_scale),
                ),
            )
        else:
            text_label.configure(
                text=f"{(self._curr_image_num % len(self._data_handlers)) + 1}"
                f"/{len(self._data_handlers)}"
            )

    def y_uncertainty(self, xval):
        # simple propagation of uncertainty with first-order
        # finite difference approximation of parameter derivatives
        par_derivs = []
        for idx, arg in enumerate(self.current_args[:]):
            shifted_args = self.current_model.args.copy()
            shifted_args[idx] = arg + abs(arg) / 1e5
            shifted_model = self.current_model.copy()
            shifted_model.args = shifted_args
            _X0, _Y0 = self.data_handler.X0, self.data_handler.Y0
            if self.data_handler.logx_flag and self.data_handler.logy_flag:
                y2 = shifted_model.eval_at(xval, X0=_X0, Y0=_Y0)
                y1 = self.current_model.eval_at(xval, X0=_X0, Y0=_Y0)
                par_derivs.append((y2 - y1) / (abs(arg) / 1e5))
            elif self.data_handler.logx_flag:
                y2 = shifted_model.eval_at(xval, X0=_X0)
                y1 = self.current_model.eval_at(xval, X0=_X0)
                par_derivs.append((y2 - y1) / (abs(arg) / 1e5))
            elif self.data_handler.logy_flag:
                y2 = shifted_model.eval_at(xval, Y0=_Y0)
                y1 = self.current_model.eval_at(xval, Y0=_Y0)
                par_derivs.append((y2 - y1) / (abs(arg) / 1e5))
            else:
                y2 = shifted_model.eval_at(xval)
                y1 = self.current_model.eval_at(xval)
                par_derivs.append((y2 - y1) / (abs(arg) / 1e5))

        partial_V_partial = 0
        for i, _ in enumerate(self.current_args):
            for j, _ in enumerate(self.current_args):
                partial_V_partial += par_derivs[i] * self.current_covariance[i, j] * par_derivs[j]
        return np.sqrt(partial_V_partial)

    def save_show_fit_image(self):

        plot_model = self.current_model.copy()

        handler = self.data_handler

        x_points = handler.unlogged_x_data
        y_points = handler.unlogged_y_data
        sigma_x_points = handler.unlogged_sigmax_data
        sigma_y_points = handler.unlogged_sigmay_data
        upper_bar = [y + dy for y, dy in zip(y_points, sigma_y_points)]
        lower_bar = [y - dy for y, dy in zip(y_points, sigma_y_points)]

        smooth_x_for_fit = np.linspace(min(x_points), max(x_points), 4 * len(x_points))

        if handler.logx_flag and handler.logy_flag:
            fit_vals = [
                plot_model.eval_at(xi, X0=self.data_handler.X0, Y0=self.data_handler.Y0)
                for xi in smooth_x_for_fit
            ]
        elif handler.logx_flag:
            fit_vals = [plot_model.eval_at(xi, X0=self.data_handler.X0) for xi in smooth_x_for_fit]
        elif handler.logy_flag:
            fit_vals = [plot_model.eval_at(xi, Y0=self.data_handler.Y0) for xi in smooth_x_for_fit]
        else:
            fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]

        plt.close()
        plt.figure(
            facecolor=self._bg_color,
            figsize=(6.4 * self._image_r, 4.8 * self._image_r),
            dpi=100 + int(np.log10(len(x_points))),
        )
        plt.errorbar(
            x_points,
            y_points,
            xerr=sigma_x_points,
            yerr=sigma_y_points,
            fmt="o",
            color=self._dataaxes_color,
        )

        plt.plot(smooth_x_for_fit, fit_vals, "-", color=self._fit_color)
        if self._show_error_bands in [1, 3]:
            unc_list = [self.y_uncertainty(xi) for xi in smooth_x_for_fit]
            upper_error_vals = [val + unc for val, unc in zip(fit_vals, unc_list)]
            lower_error_vals = [val - unc for val, unc in zip(fit_vals, unc_list)]

            plt.fill_between(
                smooth_x_for_fit,
                lower_error_vals,
                upper_error_vals,
                color=self._fit_color,
                alpha=0.5,
            )

        if self._show_error_bands in [2, 3]:
            unc_list = [self.y_uncertainty(xi) for xi in smooth_x_for_fit]
            upper_2error_vals = [val + 2 * unc for val, unc in zip(fit_vals, unc_list)]
            lower_2error_vals = [val - 2 * unc for val, unc in zip(fit_vals, unc_list)]

            plt.fill_between(
                smooth_x_for_fit,
                lower_2error_vals,
                upper_2error_vals,
                color=self._fit_color,
                alpha=0.5,
            )

        plt.xlabel(handler.x_label)
        plt.ylabel(handler.y_label)
        axes: plt.axes = plt.gca()
        axes.tick_params(color=self._dataaxes_color, labelcolor=self._dataaxes_color)
        axes.xaxis.label.set_color(self._dataaxes_color)
        axes.yaxis.label.set_color(self._dataaxes_color)
        for spine in axes.spines.values():
            spine.set_edgecolor(self._dataaxes_color)

        pf.zero_out_axes(axes)

        if handler.logx_flag:
            pf.set_xaxis_format_log(axes, x_points)
        else:
            pf.set_xaxis_format_linear(axes, x_points)

        if handler.logy_flag:
            pf.set_yaxis_format_log(axes, y_points)
        else:
            pf.set_yaxis_format_linear(axes, y_points)

        axes.set_facecolor(self._bg_color)

        pf.fix_axes_labels(
            axes,
            min(x_points),
            max(x_points),
            min(lower_bar),
            max(upper_bar),
            self.data_handler.x_label,
        )
        plt.savefig(self._image_path, facecolor=self._bg_color)

        # change the view to show the fit as well
        self.switch_image()
        self._showing_fit_image = True
        self._showing_fit_all_image = False

    def save_show_fit_all(self, args_list):

        plot_model = self.optimizer.shown_model.copy()

        num_sets = len(self._data_handlers)
        abs_min_x, abs_min_y = 1e5, 1e5
        abs_max_x, abs_max_y = -1e5, -1e5

        sum_len = 0

        plt.close()
        plt.figure(
            facecolor=self._bg_color,
            figsize=(6.4 * self._image_r, 4.8 * self._image_r),
            dpi=100 + int(np.log10(len(self.data_handler.unlogged_x_data))),
        )
        axes: plt.axes = plt.gca()
        axes.tick_params(color=self._dataaxes_color, labelcolor=self._dataaxes_color)
        axes.xaxis.label.set_color(self._dataaxes_color)
        axes.yaxis.label.set_color(self._dataaxes_color)
        for spine in axes.spines.values():
            spine.set_edgecolor(self._dataaxes_color)
        for idx, (handler, args) in enumerate(zip(self._data_handlers, args_list)):

            x_points = handler.unlogged_x_data
            y_points = handler.unlogged_y_data
            sigma_x_points = handler.unlogged_sigmax_data
            sigma_y_points = handler.unlogged_sigmay_data

            sum_len += len(x_points)
            smooth_x_for_fit = np.linspace(x_points[0], x_points[-1], 4 * len(x_points))
            plot_model.args = args
            logger(f"{plot_model.args=}")
            if handler.logx_flag and handler.logy_flag:
                fit_vals = [
                    plot_model.eval_at(xi, X0=handler.X0, Y0=handler.Y0) for xi in smooth_x_for_fit
                ]
            elif handler.logx_flag:
                fit_vals = [plot_model.eval_at(xi, X0=handler.X0) for xi in smooth_x_for_fit]
            elif handler.logy_flag:
                fit_vals = [plot_model.eval_at(xi, Y0=handler.Y0) for xi in smooth_x_for_fit]
            else:
                fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]

            # col = 255 ** (idx/num_sets) / 255
            # col = np.sqrt(idx / num_sets)
            col_tuple = [
                (icol / max(self._dataaxes_color) if max(self._dataaxes_color) > 0 else 1)
                * (idx / num_sets)
                for icol in self._dataaxes_color
            ]
            axes.errorbar(
                x_points,
                y_points,
                xerr=sigma_x_points,
                yerr=sigma_y_points,
                fmt="o",
                color=col_tuple,
            )
            plt.plot(smooth_x_for_fit, fit_vals, "-", color=col_tuple)

            min_x, max_x = min(x_points), max(x_points)
            min_y, max_y = min(y_points), max(y_points)

            abs_min_x = min(abs_min_x, min_x)
            abs_min_y = min(abs_min_y, min_y)
            abs_max_x = max(abs_max_x, max_x)
            abs_max_y = max(abs_max_y, max_y)

            plt.draw()

        # also add average fit
        smooth_x_for_fit = np.linspace(abs_min_x, abs_max_x, sum_len)
        if self.data_handler.logx_flag and self.data_handler.logy_flag:
            fit_vals = [
                self.optimizer.shown_model.eval_at(
                    xi, X0=self.data_handler.X0, Y0=self.data_handler.Y0
                )
                for xi in smooth_x_for_fit
            ]
        elif self.data_handler.logx_flag:
            fit_vals = [
                self.optimizer.shown_model.eval_at(xi, X0=self.data_handler.X0)
                for xi in smooth_x_for_fit
            ]
        elif self.data_handler.logy_flag:
            fit_vals = [
                self.optimizer.shown_model.eval_at(xi, Y0=self.data_handler.Y0)
                for xi in smooth_x_for_fit
            ]
        else:
            fit_vals = [self.optimizer.shown_model.eval_at(xi) for xi in smooth_x_for_fit]

        plt.plot(smooth_x_for_fit, fit_vals, "-", color=self._fit_color)
        if self._show_error_bands in [1, 3]:
            unc_list = [self.y_uncertainty(xi) for xi in smooth_x_for_fit]
            upper_error_vals = [val + unc for val, unc in zip(fit_vals, unc_list)]
            lower_error_vals = [val - unc for val, unc in zip(fit_vals, unc_list)]

            plt.plot(smooth_x_for_fit, upper_error_vals, "--", color=self._fit_color)
            plt.plot(smooth_x_for_fit, lower_error_vals, "--", color=self._fit_color)
        if self._show_error_bands in [2, 3]:
            unc_list = [self.y_uncertainty(xi) for xi in smooth_x_for_fit]
            upper_2error_vals = [val + 2 * unc for val, unc in zip(fit_vals, unc_list)]
            lower_2error_vals = [val - 2 * unc for val, unc in zip(fit_vals, unc_list)]

            plt.plot(smooth_x_for_fit, upper_2error_vals, ":", color=self._fit_color)
            plt.plot(smooth_x_for_fit, lower_2error_vals, ":", color=self._fit_color)

        for spine in axes.spines.values():
            spine.set_color(self._dataaxes_color)

        plt.xlabel(self.data_handler.x_label)
        plt.ylabel(self.data_handler.y_label)

        pf.zero_out_axes(axes)

        if self.data_handler.logx_flag:
            pf.set_xaxis_format_log(axes, xminmax=(abs_min_x, abs_max_x))
        else:
            pf.set_xaxis_format_linear(axes, xminmax=(abs_min_x, abs_max_x))
        if self.data_handler.logy_flag:
            pf.set_yaxis_format_log(axes, yminmax=(abs_min_y, abs_max_y))
        else:
            pf.set_yaxis_format_linear(axes, yminmax=(abs_min_y, abs_max_y))
        axes.set_facecolor(self._bg_color)

        pf.fix_axes_labels(
            axes,
            abs_min_x,
            abs_max_x,
            abs_min_y,
            abs_max_y,
            self.data_handler.x_label,
        )
        plt.savefig(self._image_path, facecolor=self._bg_color)

        # change the view to show the fit as well
        self.switch_image()
        self._showing_fit_image = True
        self._showing_fit_all_image = True

    def show_current_data_with_fit(self, quiet=False, do_halving=False, show_image=True):
        # this fits the data again, unlike save_show_fit
        self.update_optimizer()

        if (
            self._model_name_tkstr.get() == "Gaussian"
            and self.current_model.name == "Normal"
            and not self.data_handler.normalized
        ):
            self.fit_data_command()
            return
        if (
            self._model_name_tkstr.get() == "Gaussian"
            and self.current_model.name == "Gaussian"
            and self.data_handler.normalized
        ):
            self.fit_data_command()
            return

        # changes optimizer's _shown variables
        self.optimizer.fit_this_and_get_model_and_covariance(
            self.current_model, do_halving=do_halving
        )

        if not quiet:
            self.add_message(f"\n \n> For {self.data_handler.shortpath} \n")
            self.print_results_to_console()
        if show_image:
            self.save_show_fit_image()

    # POLYNOMIAL frame ---------------------------------------------------------------------------->
    def create_degree_up_down_buttons(self):
        if self._new_user_stage % 19 == 0:
            return
        self._new_user_stage *= 19

        # polynomial
        self._polynomial_degree_label = tk.Label(
            master=self._polynomial_frame,
            text=f"Degree: {self._polynomial_degree_tkint.get()}",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bg=self._sbf,
        )
        down_button = tk.Button(
            self._gui.children["!frame2"].children["!frame6"],
            text=self.sym_down,
            bd=self._platform_border,
            command=self.degree_down_command,
        )
        up_button = tk.Button(
            self._gui.children["!frame2"].children["!frame6"],
            text=self.sym_up,
            bd=self._platform_border,
            command=self.degree_up_command,
        )

        self._polynomial_degree_label.grid(row=0, column=0, sticky="w")
        down_button.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsw")
        up_button.grid(row=0, column=2, pady=5, sticky="nsw")

    def hide_degree_buttons(self):
        if self._new_user_stage % 19 != 0:
            return
        self._polynomial_frame.grid_forget()

    def show_degree_buttons(self):
        if self._new_user_stage % 19 != 0:
            self.create_degree_up_down_buttons()
            return
        self._polynomial_frame.grid(row=4, column=0, sticky="w")

    def degree_down_command(self):
        if self._polynomial_degree_tkint.get() > 0:
            self._polynomial_degree_tkint.set(self._polynomial_degree_tkint.get() - 1)
        else:
            self.add_message("\n \n> Polynomials must have a degree of at least 0\n")
            return
        self._polynomial_degree_label.configure(
            text=f"Degree: {self._polynomial_degree_tkint.get()}"
        )

        if self._showing_fit_image:
            if self._refit_on_click:
                self.fit_data_command()

    def degree_up_command(self):
        if self._polynomial_degree_tkint.get() < self.max_poly_degree():
            self._polynomial_degree_tkint.set(self._polynomial_degree_tkint.get() + 1)
        else:
            self.add_message(
                f"\n \n> Degree greater than {self._polynomial_degree_tkint.get()}"
                f" will lead to an overfit."
            )
            return
        self._polynomial_degree_label.configure(
            text=f"Degree: {self._polynomial_degree_tkint.get()}"
        )

        if self._showing_fit_image:
            if self._refit_on_click:
                self.fit_data_command()

    def max_poly_degree(self):
        return len({datum.pos for datum in self.data_handler.data}) - 1

    # GAUSSIAN frame ------------------------------------------------------------------------------>
    def create_modal_up_down_buttons(self):
        if self._new_user_stage % 47 == 0:
            return
        self._new_user_stage *= 47

        # gaussian
        self._gaussian_modal_label = tk.Label(
            master=self._gaussian_frame,
            text=f"Modes: {self._gaussian_modal_tkint.get()}",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bg=self._sbf,
        )
        down_button = tk.Button(
            self._gaussian_frame,
            text=self.sym_down,
            bd=self._platform_border,
            command=self.modal_down_command,
        )
        up_button = tk.Button(
            self._gaussian_frame,
            text=self.sym_up,
            bd=self._platform_border,
            command=self.modal_up_command,
        )

        self._gaussian_modal_label.grid(row=0, column=0, sticky="w")
        down_button.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="w")
        up_button.grid(row=0, column=2, pady=5, sticky="w")

    def hide_modal_buttons(self):
        if self._new_user_stage % 47 != 0:
            return
        self._gaussian_frame.grid_forget()

    def show_modal_buttons(self):
        if self._new_user_stage % 47 != 0:
            self.create_modal_up_down_buttons()
            return
        self._gaussian_frame.grid(row=4, column=0, sticky="w")

    def modal_down_command(self):
        if self._gaussian_modal_tkint.get() > 1:
            self._gaussian_modal_tkint.set(self._gaussian_modal_tkint.get() - 1)
            self._gaussian_modal_label.configure(text=f"Modes: {self._gaussian_modal_tkint.get()}")
        else:
            self.add_message("> Gaussians models must have at least 1 peak\n")

    def modal_up_command(self):
        if self._gaussian_modal_tkint.get() < self.max_modal():
            self._gaussian_modal_tkint.set(self._gaussian_modal_tkint.get() + 1)
            self._gaussian_modal_label.configure(text=f"Modes: {self._gaussian_modal_tkint.get()}")
        else:
            self.add_message(
                f"> Multi-modal Gaussian models with "
                f"{len({datum.pos for datum in self.data_handler.data})} "
                f"x-positions can have at most {self.max_modal()} peaks.\n"
            )

    def max_modal(self):
        return len({datum.pos for datum in self.data_handler.data}) // 3

    # PROCEDURAL frame ---------------------------------------------------------------------------->
    def create_procedural_options(self):
        if self._new_user_stage % 31 == 0:
            return
        self._new_user_stage *= 31

        self.create_default_checkboxes()
        self.create_depth_up_down_buttons()

    def hide_procedural_options(self):
        if self._new_user_stage % 31 != 0:
            return
        self._procedural_frame.grid_forget()

    def show_procedural_options(self):
        if self._new_user_stage % 31 != 0:
            self.create_procedural_options()
            return
        self._procedural_frame.grid(row=4, column=0, sticky="w")

    def create_default_checkboxes(self):

        if self._new_user_stage % 31 != 0:
            return
        # still to add:
        # fit data / vs / search models on Procedural
        # sliders for initial parameter guesses

        for idx, name in enumerate(self._checkbox_names_list):
            my_font = "TkDefaultFont", int(12 * self._default_os_scaling * self._platform_scale)
            # logger(regex.split(f" ", self._custom_function_names))
            checkbox = tk.Checkbutton(
                master=self._procedural_frame,
                text=name,
                variable=self._use_func_dict_name_tkbool[name],
                onvalue=True,
                offvalue=False,
                font=my_font,
                command=self.checkbox_on_off_command,
                bg=self._sbf,
            )
            checkbox.grid(
                row=idx % (len(self._checkbox_names_list) - 1),
                column=2 * ((idx + 1) // len(self._checkbox_names_list)),
                sticky="w",
            )
            if idx == len(self._checkbox_names_list) - 1:
                self._custom_checkbox = checkbox

        self.update_custom_checkbox()
        # self.create_custom_remove_menu()

    def update_custom_checkbox(self):
        if self._new_user_stage % 31 != 0:
            return
        self._custom_binding = self._custom_checkbox.bind(
            self._right_click, self.do_custom_remove_popup
        )
        self._custom_checkbox.configure(
            text="custom: "
            + ", ".join([x for x in regex.split(" ", self._custom_function_names) if x])
        )
        self.create_custom_remove_menu()

    def checkbox_on_off_command(self):
        logger("Activated re-build of composite list")
        self._changed_optimizer_opts_flag = True

    def create_depth_up_down_buttons(self):
        # duplication taken care of with % 31 i.e. default_checkboxes
        self._depth_label = tk.Label(
            master=self._procedural_frame,
            text=f"Depth: {self._max_functions_tkint.get()}",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bg=self._sbf,
        )
        down_button = tk.Button(
            self._procedural_frame,
            text=self.sym_down,
            bd=self._platform_border,
            command=self.depth_down_command,
        )
        up_button = tk.Button(
            self._procedural_frame,
            text=self.sym_up,
            bd=self._platform_border,
            command=self.depth_up_command,
        )

        self._depth_label.grid(row=100, column=0, sticky="w")
        down_button.grid(row=100, column=1, padx=(5, 0), pady=5, sticky="w")
        up_button.grid(row=100, column=2, pady=5, sticky="w")

    def depth_down_command(self):
        if self._max_functions_tkint.get() > 1:
            self._max_functions_tkint.set(self._max_functions_tkint.get() - 1)
        else:
            self.add_message("> Must have a depth of at least 1\n")
        # noinspection PyTypeChecker
        self._depth_label.configure(text=f"Depth: {self._max_functions_tkint.get()}")
        self._changed_optimizer_opts_flag = True

    def depth_up_command(self):
        if self._max_functions_tkint.get() >= 7:
            self.add_message("\n \n> Cannot exceed a depth of 7\n")
        elif self._max_functions_tkint.get() >= self.max_poly_degree() + 1:
            self.add_message(
                f"\n \n> Depth greater than {self.max_poly_degree() + 1} will lead to an overfit."
            )
        else:
            self._max_functions_tkint.set(self._max_functions_tkint.get() + 1)
        # noinspection PyTypeChecker
        self._depth_label.configure(text=f"Depth: {self._max_functions_tkint.get()}")
        self._changed_optimizer_opts_flag = True

    def create_custom_remove_menu(self):

        logger("In create custom remove menu")
        head_menu = tk.Menu(master=self._gui, tearoff=0)

        names_menu = tk.Menu(master=head_menu, tearoff=0)
        names_menu.add_command(
            label="All Functions", command=partial(self.remove_named_custom, "All")
        )
        for name in [x for x in regex.split(" ", self._custom_function_names) if x]:
            logger(f"Added command for {name}")
            names_menu.add_command(label=name, command=partial(self.remove_named_custom, name))

        head_menu.add_cascade(label="Remove Custom", menu=names_menu)

        self._custom_remove_menu = head_menu

    def do_custom_remove_popup(self, event: tk.Event):
        try:
            self._custom_remove_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._custom_remove_menu.grab_release()

    def remove_named_custom(self, name: str):

        custom_names = [x for x in regex.split(" ", self._custom_function_names) if x]
        custom_forms = [x for x in regex.split(" ", self._custom_function_forms) if x]

        logger("Remove named custom debug: ")

        if name == "":
            return
        if name == "All":
            logger("Removing all custom functions")
            for iname, iform in zip(custom_names[:], custom_forms[:]):
                custom_names.remove(iname)
                custom_forms.remove(iform)
                PrimitiveFunction.built_in_dict().pop(iname, None)
        else:
            logger(f"Remove named custom {name}")
            for iname, iform in zip(custom_names[:], custom_forms[:]):
                if iname == name:
                    custom_names.remove(iname)
                    custom_forms.remove(iform)

                    self.optimizer.remove_named_from_prim_list(name)
                    PrimitiveFunction.built_in_dict().pop(iname, None)
                    break

        self._custom_function_names = " ".join(custom_names)
        self._custom_function_forms = " ".join(custom_forms)
        self._changed_optimizer_opts_flag = True
        self.save_defaults()
        self.update_optimizer()
        self.update_custom_checkbox()

        logger([name for name in regex.split(" ", self._custom_function_names) if name])

    # brute force -- also associated with fit_options panel for the pause button
    def begin_brute_loop(self):
        self._gui.update_idletasks()
        self.add_message(f"\n \n> For {self.data_handler.shortpath} \n")
        self.add_message(f"{self.optimizer.gen_idx:>10} models tested")
        self._gui.after_idle(self.maintain_brute_loop)

    def maintain_brute_loop(self):
        # TODO: add a model counter to show how many have been tested
        # should also auto-pause when red chi sqr reaches ~1. Same for procedural,
        # to avoid overfitting (the linear data is a good example of fits that becomes
        # infinitesimally better with more parameters)
        if self.brute_forcing:
            status = self.optimizer.async_find_best_model_for_dataset()
            if status != "":
                self.pause_command()
                self.add_message(f"\n\n >>> End of brute-forcing reached <<< {status}\n\n")
            # self.optimizer.show_fit(pause_on_image=True)
            if self.current_rchisqr != self._optimizer.top5_rchisqrs[0]:
                self.make_top_shown()
                self.save_show_fit_image()
                if self.current_rchisqr < 0.001:
                    self.add_message(f"\n \n >>> Perfect fit found <<< {status}\n\n")
                    self.pause_command()
            self.update_top5_dropdown()
            self.update_number_tested()
            self._gui.after(1, self.maintain_brute_loop)

        self.update_pause_button()

    def create_pause_button(self):
        if self._new_user_stage % 37 == 0:
            return
        self._new_user_stage *= 37

        self._pause_button = tk.Button(
            self._fit_options_frame,
            text="Pause",
            width=6 - self._platform_offset,
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            bd=self._platform_border,
            command=self.pause_command,
        )
        self._pause_button.grid(row=0, column=2, padx=(5, 0), sticky="nw")

    def hide_pause_button(self):
        if self._new_user_stage % 37 != 0:
            return
        self._pause_button.grid_forget()

    def show_pause_button(self):
        if self._new_user_stage % 37 != 0:
            self.create_pause_button()
            return
        self._pause_button.grid(row=0, column=2, padx=(5, 0), pady=5, sticky="w")
        self.update_pause_button()

    def update_pause_button(self):
        if self._new_user_stage % 37 != 0:
            return

        if self.brute_forcing:
            self._pause_button.configure(text="Pause")
        else:
            self._pause_button.configure(text="Go")

    def pause_command(self):
        if self.brute_forcing:
            self.brute_forcing = False
            self.add_message(f"\n \n> For {self.data_handler.shortpath} \n")
            self.print_results_to_console()
        else:
            self.brute_forcing = True
            # it's actually a go button
            self.begin_brute_loop()

    def update_number_tested(self):
        self._progress_label.configure(text=f"{self.optimizer.gen_idx:>10} models tested")

    # manual model
    def create_manual_fields(self):
        if self._new_user_stage % 53 == 0:
            return
        self._new_user_stage *= 53

        def_font = (
            "TkDefaultFont",
            int(12 * self._default_os_scaling * self._platform_scale),
        )

        name_label = tk.Label(
            master=self._manual_frame,
            text="Function's Name",
            font=def_font,
            bg=self._sbf,
        )
        name_label.grid(row=0, column=0, sticky="w")
        name_data = tk.Entry(master=self._manual_frame, width=30, font=def_font)
        name_data.insert(
            0,
            (
                "ManualEntryFunc"
                if self._default_manual_name == "N/A"
                else self._default_manual_name
            ),
        )
        name_data.grid(row=0, column=1, sticky="w")

        # form_label = tk.Label(master=self._manual_frame, text="Function's Form")
        # form_label.grid(row=1, column=0, sticky='w')
        # form_data = tk.Entry(master=self._manual_frame, width=60)
        # form_data.insert(0, "sin(pow1)+sin(pow1)+sin(pow1)+sin(pow1)")
        # form_data.grid(row=1, column=1, sticky='w')

        long_label = tk.Label(
            master=self._manual_frame,
            text="Function's Form",
            font=def_font,
            bg=self._sbf,
        )
        long_label.grid(row=1, column=0, sticky="nw")
        long_data = tk.Text(master=self._manual_frame, width=55, height=5, font=def_font)
        long_data.insert(
            "1.0",
            (
                "logistic(pow1+pow0)"
                if self._default_manual_form == "N/A"
                else self._default_manual_form
            ),
        )
        long_data.grid(row=1, column=1, sticky="w")

        self._error_label = tk.Label(master=self._manual_frame, text="", fg="#EF0909", bg=self._sbf)
        self._error_label.grid(row=2, column=1, sticky="w", pady=5)

        current_name_title_label = tk.Label(
            master=self._manual_frame,
            text="Current Name:",
            font=def_font,
            bg=self._sbf,
        )
        current_name_title_label.grid(row=3, column=0, sticky="w", pady=(5, 0))
        self._current_name_label = tk.Label(master=self._manual_frame, text="N/A", bg=self._sbf)
        self._current_name_label.grid(row=3, column=1, sticky="w", pady=(5, 0))
        current_form_title_label = tk.Label(
            master=self._manual_frame,
            text="Current Form:",
            font=def_font,
            bg=self._sbf,
        )
        current_form_title_label.grid(row=4, column=0, sticky="w")
        self._current_form_label = tk.Label(master=self._manual_frame, text="N/A", bg=self._sbf)
        self._current_form_label.grid(row=4, column=1, sticky="w")

        submit_button = tk.Button(
            master=self._manual_frame,
            text="Validate",
            width=len("Validate") - self._platform_offset,
            font=def_font,
            bd=self._platform_border,
            command=self.validate_manual_function_command,
        )
        submit_button.grid(row=1, column=10, padx=5, pady=0, sticky="s")
        self.create_library_options()

        if self._default_manual_name != "N/A" and self._optimizer is not None:
            logger(self._default_manual_form)
            manual_model = CompositeFunction.construct_model_from_str(
                form=self._default_manual_form,
                error_handler=self.add_message,
                name=self._default_manual_name,
            )
            if manual_model is None:
                return

            self._current_name_label.configure(text=self._default_manual_name, font=def_font)
            self._current_form_label.configure(text=self._default_manual_form, font=def_font)
            manual_model.print_tree()
            self._manual_model = manual_model

    def hide_manual_fields(self):
        if self._new_user_stage % 53 != 0:
            return
        self._manual_frame.grid_forget()
        self.hide_library_options()
        self.hide_sliders()

    def show_manual_fields(self):
        if self._new_user_stage % 53 != 0:
            self.create_manual_fields()
            return

        self._manual_frame.grid(row=4, column=0, sticky="w")
        self.show_library_options()

    def validate_manual_function_command(self) -> bool:
        # pylint: disable=too-many-return-statements, too-many-branches
        self.add_message(
            f"\n \n> Validating {self._manual_frame.children['!entry'].get()} with form\n"
            f"  {self._manual_frame.children['!text'].get('1.0', tk.END)}"
        )

        namestr = "".join(self._manual_frame.children["!entry"].get().split())
        formstr = "".join(self._manual_frame.children["!text"].get("1.0", "end-1c").split())
        if "," in formstr:
            return self.error_handling(
                "Support for special functions with parameters are not yet implemented."
            )
        if "-" in formstr:
            return self.error_handling("Subtraction '-' is not a valid symbol.")
        if "/" in formstr:
            return self.error_handling("Division '/' is not a valid symbol.")
        if "^" in formstr:
            return self.error_handling("Exponentiation through '^' is not permitted.")
        if "." in formstr:
            return self.error_handling("Subpackage functions (with '.') are not permitted.")
        if ")(" in formstr:
            return self.error_handling(
                "Implicit multiplication with ')(' is not supported. Use '*' or '·'."
            )
        if "**" in formstr:
            return self.error_handling("You may not use '**' to indicate powers.")
        invalid_sequences = [
            "++",
            "+)",
            "+*",
            "+·",
            "*+",
            "*)",
            "*·",
            "·+",
            "·)",
            "·*",
            "··",
            "(+",
            "()",
            "(*",
            "(·",
            "((",
        ]
        for seq in invalid_sequences:
            if seq in formstr:
                return self.error_handling(f"Invalid sequence '{seq}'")
        open_paren = 0
        for c in formstr:
            if c == "(":
                open_paren += 1
            elif c == ")":
                open_paren -= 1
            if open_paren < 0:
                self.add_message("\n \n> Mismatched parentheses ().")
                return False
        if open_paren != 0:
            self.add_message("\n \n> Mismatched parentheses.")
            return False
        if formstr[0] in ["·", "+", ")", "*"]:
            self.error_handling(f"You can't start a function with an operation '{formstr[0]}'.")

        manual_model = CompositeFunction.construct_model_from_str(
            form=formstr,
            error_handler=self.error_handling,
            name=self._default_manual_name,
        )
        if manual_model is None:
            return False

        self._default_manual_name = namestr
        self._default_manual_form = formstr
        self._current_name_label.configure(text=self._default_manual_name)
        self._current_form_label.configure(text=self._default_manual_form)
        manual_model.print_tree()
        self.add_message(
            f"\n \n  Successfully validated! Tree below: \n{manual_model.tree_as_string()}"
        )
        logger(manual_model.tree_as_string_with_dimensions())
        self._manual_model = manual_model
        self.save_defaults()

        return True

    def error_handling(self, error_msg: str) -> bool:
        self._error_label.configure(text=error_msg)
        return False

    def create_library_options(self):
        if self._new_user_stage % 59 == 0:
            return
        self._new_user_stage *= 59

        common_width = 8 - self._platform_offset

        self._library_numpy = tk.Button(
            self._fit_options_frame,
            text="<numpy>",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            width=common_width,
            bd=self._platform_border,
            command=self.print_numpy_library,
        )
        self._library_numpy.grid(row=0, column=1, padx=(5, 0), sticky="w")

        self._library_special = tk.Button(
            self._fit_options_frame,
            text="<special>",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            width=common_width,
            bd=self._platform_border,
            command=self.print_special_library,
        )
        self._library_special.grid(row=0, column=2, sticky="w")

        self._library_stats = tk.Button(
            self._fit_options_frame,
            text="<stats>",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            width=common_width,
            bd=self._platform_border,
            command=self.print_stats_library,
        )
        self._library_stats.grid(row=0, column=3, sticky="w")

        self._library_autofit = tk.Button(
            self._fit_options_frame,
            text="<autofit>",
            font=(
                "TkDefaultFont",
                int(12 * self._default_os_scaling * self._platform_scale),
            ),
            width=common_width,
            bd=self._platform_border,
            command=self.print_autofit_library,
        )
        self._library_autofit.grid(row=0, column=4, sticky="w")

    def hide_library_options(self):
        if self._new_user_stage % 59 != 0:
            return
        self._library_numpy.grid_forget()
        self._library_special.grid_forget()
        self._library_stats.grid_forget()
        # self._library_math.grid_forget()
        self._library_autofit.grid_forget()

    def show_library_options(self):
        if self._new_user_stage % 59 != 0:
            self.create_library_options()
            return
        self._library_numpy.grid(row=0, column=1, padx=(5, 0), sticky="w")
        self._library_special.grid(row=0, column=2, sticky="w")
        self._library_stats.grid(row=0, column=3, sticky="w")
        # self._library_math.grid(row=0, column=4, sticky='w')
        self._library_autofit.grid(row=0, column=4, sticky="w")

    def print_numpy_library(self):
        buffer = "\n \n  <numpy> options: \n  "
        for memb in dir(np):
            fn: np.ufunc = getattr(np, memb)
            if (
                type(fn) is np.ufunc  # pylint: disable=unidiomatic-typecheck
                and fn.nin == 1
                and "f->f" in fn.types
            ):
                try:
                    y = fn(np.pi / 4)
                except TypeError:
                    logger(f"{memb} not 1D")
                except ValueError:
                    logger(f"{memb} doesn't accept float values")
                else:
                    logger(memb, y)
                    buffer += f"{memb}, "
            if len(buffer) > 50:
                self.add_message(buffer[:-2])
                buffer = "  "
        self.add_message(buffer[:-2])

    def print_special_library(self):
        buffer = "\n \n  <scipy.special> options: \n  "
        for memb in dir(scipy.special):
            fn = getattr(scipy.special, memb)
            if (
                type(fn) is np.ufunc  # pylint: disable=unidiomatic-typecheck
                and fn.nin == 1
                and "f->f" in fn.types
            ):
                try:
                    y = fn(np.pi / 4)
                except TypeError:
                    logger(f"{memb} not 1D")
                except ValueError:
                    logger(f"{memb} doesn't accept float values")
                else:
                    logger(memb, y)
                    buffer += f"{memb}, "
            if len(buffer) > 50:
                self.add_message(buffer[:-2])
                buffer = "  "
        self.add_message(buffer[:-2])

    def print_stats_library(self):
        buffer = "\n \n  <scipy.stats> options: \n  "
        for memb in dir(scipy.stats._continuous_distns):  # pylint: disable=protected-access
            fn = getattr(scipy.stats._continuous_distns, memb)  # pylint: disable=protected-access
            target_type = "<class 'scipy.stats."
            if str(type(fn))[:20] == target_type:  # pylint: disable=unidiomatic-typecheck
                try:
                    y = fn.pdf(np.pi / 4)
                except TypeError:
                    logger(f"{memb} not 1D")
                except ValueError:
                    logger(f"{fn} should be ok?")
                else:
                    logger(memb, y)
                    buffer += f"{memb}, "

                # Maybe add this is the next version
                # try:
                #     y = fn.cdf(np.pi / 4)
                # except TypeError:
                #     logger(f"{memb}_cdf not 1D")
                # except ValueError:
                #     logger(f"{fn} should be ok?")
                # else:
                #     logger(memb, y)
                #     buffer += f"{memb}_cdf, "
            if len(buffer) > 50:
                self.add_message(buffer[:-2])
                buffer = "  "
        self.add_message(buffer[:-2])

    def print_autofit_library(self):
        buffer = "\n \n  <autofit> options: \n  "
        for prim in PrimitiveFunction.build_built_in_dict().values():
            logger(prim.name)
            buffer += f"{prim.name}, "
            if len(buffer) > 50:
                self.add_message(buffer[:-2])
                buffer = "  "
        self.add_message(buffer[:-2])

    def create_sliders(self):
        if self._new_user_stage % 71 == 0:
            return
        if self.current_model is None:
            return
        self._new_user_stage *= 71

        self._slider_frame = tk.Frame(self._manual_frame, bg=self._sbf)
        self._slider_frame.grid(row=5, column=1, sticky="w")

        def_font = (
            "TkDefaultFont",
            int(12 * self._default_os_scaling * self._platform_scale),
        )

        # TODO: why do sliders conform to the block shape in the row above?

        for idx, _ in enumerate(self.current_args):
            new_label = tk.Label(self._slider_frame, text=f"c{idx}", font=def_font, bg=self._sbf)
            new_slider = tk.Scale(
                self._slider_frame,
                from_=10,
                to=-10,
                resolution=0.01,
                orient=tk.VERTICAL,
                command=partial(self.observe_sliders, idx),
            )

            new_slider.set(self.arg_to_slider(self.current_model.args[idx]))
            new_label.grid(row=5, column=idx, sticky="w")
            new_slider.grid(row=6, column=idx, sticky="w")
            self._slider_labels.append(new_label)
            self._sliders.append(new_slider)

    def show_sliders(self):
        if self._new_user_stage % 71 != 0:
            # self.create_sliders()
            return

        # TODO: number of sliders doesn't change when dof changes

        for idx, (slider, slider_label) in enumerate(zip(self._sliders, self._slider_labels)):
            slider_label.grid(row=5, column=idx, sticky="w")
            slider.grid(row=6, column=idx, sticky="w")

    def update_sliders(self):
        if self._new_user_stage % 71 != 0:
            return

        # self._slider_frame.grid_forget()
        for slider in self._sliders:
            slider.destroy()
        self._slider_frame.destroy()
        self._new_user_stage /= 71
        self._sliders = []
        self._slider_labels = []
        self.create_sliders()

    def hide_sliders(self):
        if self._new_user_stage % 71 != 0:
            return
        for slider, slider_label in zip(self._sliders, self._slider_labels):
            slider.grid_forget()
            slider_label.grid_forget()

    # noinspection PyUnusedLocal
    def observe_sliders(self, *args):  # pylint: disable=unused-argument
        self.current_model.args = self.get_slider_args()

        if not self._already_queued:
            self._gui.after(500, self.reshow)
            self._already_queued = True

    def get_slider_args(self) -> Union[None, list[float]]:
        if len(self._sliders) == 0:
            return None
        args_to_set = []
        for slider in self._sliders:
            arg = slider.get()
            if arg == 0:
                args_to_set.append(0)
                continue
            args_to_set.append(np.power(10, arg - 5) if arg > 0 else -np.power(10, -arg - 5))
        return args_to_set

    @staticmethod
    def arg_to_slider(val: float) -> float:
        if val**2 < 1e-10:
            return 0
        return np.log10(val) + 5 if val > 0 else -np.log10(-val) - 5

    def reshow(self):
        self.save_show_fit_image()
        self.switch_image()
        self._already_queued = False

    """
    Right Panel
    """

    # Properties and frontend functions ----------------------------------------------------------->
    def show_current_data(self):
        self.show_data()
        self._showing_fit_image = False
        self._showing_fit_all_image = False

    def update_image(self):
        if self._showing_fit_all_image:
            self.fit_all_command()
        elif self._showing_fit_image:
            self.show_current_data_with_fit()
        else:
            if self._image_path != f"{pkg_path()}/images/splash.png":
                self.show_current_data()

    @property
    def data_handler(self):
        return self._data_handlers[self._curr_image_num]

    @property
    def brute_forcing(self):
        return self._brute_forcing_tkbool.get()

    @brute_forcing.setter
    def brute_forcing(self, val):
        self._brute_forcing_tkbool.set(val)

    def update_optimizer(self):
        if self.optimizer is None:
            self.optimizer = Optimizer(
                data=self.data_handler.data,
                use_functions_dict=self.use_functions_dict,
                max_functions=self.max_functions,
                criterion=self.criterion,
            )

            self._changed_optimizer_opts_flag = True
        if self._changed_optimizer_opts_flag:  # max depth, changed dict
            self.optimizer.update_opts(
                use_functions_dict=self.use_functions_dict,
                max_functions=self.max_functions,
            )
            if self._custom_function_forms != "":
                logger(
                    f"Update_optimizer: Including custom functions "
                    f">{self._custom_function_names}< with forms >{self._custom_function_forms}<"
                )
                for name, form in zip(
                    [x for x in regex.split(" ", self._custom_function_names) if x],
                    [x for x in regex.split(" ", self._custom_function_forms) if x],
                ):
                    info_str = self._optimizer.add_primitive_to_list(name, form)
                    if info_str != "":
                        self.add_message(f"\n \n>>> {info_str} <<<\n \n")
                        if info_str[:9] == "Corrupted":
                            self.remove_named_custom("All")
                        if info_str[:9] == "One of th":
                            self.remove_named_custom(name)
            self._changed_optimizer_opts_flag = False
        if self._changed_data_flag:
            self.optimizer.set_data_to(self.data_handler.data)
            self._changed_data_flag = False

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, other):
        self._optimizer = other

    def make_top_shown(self):
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.optimizer.shown_model = self.optimizer.top_model
            self.optimizer.shown_covariance = self.optimizer.top_cov
            self.optimizer.shown_rchisqr = self.optimizer.top_rchisqr

    @property
    def current_model(self) -> CompositeFunction:
        return self.optimizer.shown_model

    @current_model.setter
    def current_model(self, other):
        self.optimizer.shown_model = other

    @property
    def current_args(self) -> list[float]:
        return self.optimizer.shown_parameters

    @property
    def current_uncs(self) -> list[float]:
        return self.optimizer.shown_uncertainties

    @property
    def current_covariance(self) -> np.ndarray:
        return self.optimizer.shown_covariance

    @current_covariance.setter
    def current_covariance(self, other):
        self.optimizer.shown_covariance = other

    @property
    def current_rchisqr(self) -> float:
        return self.optimizer.shown_rchisqr

    @current_rchisqr.setter
    def current_rchisqr(self, other):
        self.optimizer.shown_rchisqr = other

    @property
    def use_functions_dict(self):
        return {key: tkBoolVar.get() for key, tkBoolVar in self._use_func_dict_name_tkbool.items()}

    @property
    def max_functions(self) -> int:
        return self._max_functions_tkint.get()

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, other: str):
        self._criterion = other

    def bg_color_default(self, do_update_image=True):
        self._default_bg_colour = "Default"
        self._bg_color = (112 / 255, 146 / 255, 190 / 255)
        self.checkmark_background_options(0)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def bg_color_white(self, do_update_image=True):
        self._default_bg_colour = "White"
        self._bg_color = (1.0, 1.0, 1.0)
        self.checkmark_background_options(1)
        if self._default_fit_colour == "White":  # shouldn't allow white on white
            self.fit_color_black(do_update_image=False)
        if self._default_dataaxes_colour == "White":  # shouldn't allow white on white
            self.dataaxes_color_default(do_update_image=False)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def bg_color_dark(self, do_update_image=True):
        self._default_bg_colour = "Dark"
        self._bg_color = (0.2, 0.2, 0.2)
        self.checkmark_background_options(2)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def bg_color_black(self, do_update_image=True):
        self._default_bg_colour = "Black"
        self._bg_color = (0.0, 0.0, 0.0)
        self.checkmark_background_options(3)
        if self._default_fit_colour == "Black":  # shouldn't allow black on black
            self.fit_color_white(do_update_image=False)
        if self._default_dataaxes_colour == "Default":
            self.dataaxes_color_white(do_update_image=False)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def dataaxes_color_default(self, do_update_image=True):
        self._default_dataaxes_colour = "Default"
        self._dataaxes_color = (0.0, 0.0, 0.0)
        self.checkmark_dataaxis_options(0)
        if (
            self._default_bg_colour == "Black"
        ):  # should allow black on black in case they don't want axes
            self.bg_color_white(do_update_image=False)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def dataaxes_color_white(self, do_update_image=True):
        self._default_dataaxes_colour = "White"
        self._dataaxes_color = (1.0, 1.0, 1.0)
        self.checkmark_dataaxis_options(1)
        if (
            self._default_bg_colour == "White"
        ):  # should allow white on white in case they don't want axes
            self.bg_color_black(do_update_image=False)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def fit_color_default(self, do_update_image=True):
        self._default_fit_colour = "Default"
        self._fit_color = (1.0, 0.0, 0.0)
        self.checkmark_fit_colour_options(0)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def fit_color_white(self, do_update_image=True):
        self._default_fit_colour = "White"
        self._fit_color = (1.0, 1.0, 1.0)
        self.checkmark_fit_colour_options(1)
        if self._default_bg_colour == "White":  # can't have white on white
            self.bg_color_black(do_update_image=False)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def fit_color_black(self, do_update_image=True):
        self._default_fit_colour = "Black"
        self._fit_color = (0.0, 0.0, 0.0)
        self.checkmark_fit_colour_options(2)
        if self._default_bg_colour == "Black":  # can't have black on black
            self.bg_color_white(do_update_image=False)
        if do_update_image:
            self.update_image()
            self.save_defaults()

    def console_color_default(self):
        self._default_console_colour = "Default"
        self._console_color = "black"
        self.checkmark_printout_background_options(0)
        if self._default_printout_colour == "Black":
            self.printout_color_white()
        self.save_defaults()

        self._right_panel_frame.destroy()
        self.create_right_panel()
        self.add_message("Changed console colour to default, black.")
        # self.add_message("Please restart MIW's AutoFit for these changes to take effect.")

    def console_color_white(self):
        self._default_console_colour = "White"
        self._console_color = "white"
        self.checkmark_printout_background_options(1)
        if self._default_printout_colour == "White":
            self.printout_color_black()
        self.save_defaults()
        self._right_panel_frame.destroy()
        self.create_right_panel()
        self.add_message("Changed console colour to white.")
        # self.add_message("Please restart MIW's AutoFit for these changes to take effect.")

    def console_color_pale(self):
        self._default_console_colour = "Pale"
        self._console_color = self._sbf
        self.checkmark_printout_background_options(2)
        if self._default_printout_colour == "White":
            self.printout_color_black()
        self.save_defaults()
        self._right_panel_frame.destroy()
        self.create_right_panel()
        self.add_message("Changed console colour to pale.")
        # self.add_message("Please restart MIW's AutoFit for these changes to take effect.")

    def printout_color_default(self):
        self._default_printout_colour = "Default"
        self._printout_color = (0, 200, 0)
        self.checkmark_printout_options(0)
        self.save_defaults()
        self.add_message("Changed printout colour to default, green.")

    def printout_color_white(self):
        self._default_printout_colour = "White"
        self._printout_color = (255, 255, 255)
        self.checkmark_printout_options(1)
        if self._default_console_colour == "White":
            self.console_color_default()
        self.save_defaults()
        self.add_message("Changed printout colour to white.")

    def printout_color_black(self):
        self._default_printout_colour = "Black"
        self._printout_color = (0, 0, 0)
        self.checkmark_printout_options(2)
        if self._default_console_colour == "Default":
            self.console_color_white()
        self.save_defaults()
        self.add_message("Changed printout colour to black.")

    def size_down(self):
        logger("Increasing resolution / decreasing text size")
        self._default_os_scaling -= 0.1
        self.add_message("> Size down")
        self.restart_command()

    def size_up(self):
        logger("Decreasing resolution / increasing text size")
        self._default_os_scaling += 0.1
        self.add_message("> Size up")
        self.restart_command()

    def refit_always(self):
        self._refit_on_click = True
        self.hide_refit_button()
        self.add_message("\n \n> All images now show a true fit of the selected model")
        self.checkmark_refit_options(0)
        self.save_defaults()

    def refit_sometimes(self):
        self._refit_on_click = False
        self.show_refit_button()
        self.add_message(
            "\n \n> Images now show the current model with the previous data's fit."
            "\nClick refit to see the fit to the new data."
        )
        self.checkmark_refit_options(1)
        self.save_defaults()

    def criterion_rchisqr(self):
        self.checkmark_criterion_options(0)
        self.criterion = "rchisqr"
        self.add_message(f"\n \n> Default criterion {self.sym_chi}{sup(2)}/dof selected. Here")
        self.add_message(
            f"  {self.sym_chi}{sup(2)} is defined as \U000003A3\U00001D62 "
            f"[f(x\U00001D62)-y\U00001D62]{sup(2)}/\U000003C3\U00001D62{sup(2)}"
        )
        self.add_message("  and dof = N - k = (num data points) - (number of parameters in model)")
        self.add_message(
            "  When \U000003C3\U00001D62 is not provided (no data uncertainty),\n"
            "  \U000003C3\U00001D62 is defined to be one tenth of ymax-ymin."
        )
        if self.optimizer is None:
            return
        if not self._showing_fit_image:
            return
        self.optimizer.criterion = self.optimizer.reduced_chi_squared_of_fit
        # TODO: what's the difference between update top5 chisqrs and update top5 dropwdown?
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()
            self.set_which5_no_trace(
                f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
            )

        self.save_defaults()

    def criterion_AIC(self):
        self.checkmark_criterion_options(1)
        self.criterion = "AIC"
        self.add_message("\n \n> Akaike Information Criterion (AIC) selected.")
        self.add_message(
            f"  We define AIC as {self.sym_chi}{sup(2)} + 2k\n"
            f"  where k = (number of parameters in model) "
        )
        if self.optimizer is None:
            return
        if not self._showing_fit_image:
            return
        self.optimizer.criterion = self.optimizer.akaike_criterion
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()
            self.set_which5_no_trace(
                f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
            )

        self.save_defaults()

    def criterion_AICc(self):
        self.checkmark_criterion_options(2)
        self.criterion = "AICc"
        self.add_message("\n \n> Corrected Akaike Information Criterion (AICc) selected.")
        self.add_message(
            "  AICc is defined as AIC + 2k(k+1)/(N-k-1),\n"
            "  where N is (num data points) and k = (number of parameters in model)"
        )
        if self.optimizer is None:
            return
        if not self._showing_fit_image:
            return
        self.optimizer.criterion = self.optimizer.akaike_criterion_corrected
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()
            self.set_which5_no_trace(
                f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
            )

        self.save_defaults()

    def criterion_BIC(self):
        self.checkmark_criterion_options(3)
        self.criterion = "BIC"
        self.add_message("\n \n> Bayes Information Criterion BIC selected.")
        self.add_message(
            f"  We define BIC as {self.sym_chi}{sup(2)} + k·log(N),\n"
            f"  where N is (num data points) and k = (number of parameters in model)"
        )
        if self.optimizer is None:
            return
        if not self._showing_fit_image:
            return
        self.optimizer.criterion = self.optimizer.bayes_criterion
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()
            self.set_which5_no_trace(
                f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
            )

        self.save_defaults()

    def criterion_HQIC(self):
        self.checkmark_criterion_options(4)
        logger(f"Changed to HQIC from {self.criterion}")
        self.criterion = "HQIC"
        self.add_message("\n \n> Hannan-Quinn Information Criterion HQIC selected.")
        self.add_message(
            f"  We define HQIC as {self.sym_chi}{sup(2)} + 2k·log(log(N)),\n"
            f"  where N is (num data points) and k = (number of parameters in model)"
        )
        if self.optimizer is None:
            return
        if not self._showing_fit_image:
            return
        self.optimizer.criterion = self.optimizer.hannan_quinn_criterion
        if self._model_name_tkstr.get() in ["Procedural", "Brute-Force"]:
            self.update_top5_chisqrs()
            self.set_which5_no_trace(
                f"{self.optimizer.criterion(self.current_model):.2F}: {self.current_model.name}"
            )

        self.save_defaults()

    def checkmark_background_options(self, idx: int):
        if idx < 0 or idx > len(self._background_labels):
            return
        for i, label in enumerate(self._background_labels):
            self._background_menu.entryconfigure(
                i, label=label + (self.sym_check if i == idx else "")
            )

    def checkmark_dataaxis_options(self, idx: int):
        if idx < 0 or idx > len(self._dataaxis_labels):
            return
        for i, label in enumerate(self._dataaxis_labels):
            self._dataaxis_menu.entryconfigure(
                i, label=label + (self.sym_check if i == idx else "")
            )

    def checkmark_fit_colour_options(self, idx: int):
        if idx < 0 or idx > len(self._fit_colour_labels):
            return
        for i, label in enumerate(self._fit_colour_labels):
            self._fit_colour_menu.entryconfigure(
                i, label=label + (self.sym_check if i == idx else "")
            )

    def checkmark_printout_background_options(self, idx: int):
        if idx < 0 or idx > len(self._printout_background_labels):
            return
        for i, label in enumerate(self._printout_background_labels):
            self._printout_background_menu.entryconfigure(
                i, label=label + (self.sym_check if i == idx else "")
            )

    def checkmark_printout_options(self, idx: int):
        if idx < 0 or idx > len(self._printout_labels):
            return
        for i, label in enumerate(self._printout_labels):
            self._printout_menu.entryconfigure(
                i, label=label + (self.sym_check if i == idx else "")
            )

    def checkmark_refit_options(self, idx: int):
        if idx < 0 or idx > len(self._refit_labels):
            return
        for i, label in enumerate(self._refit_labels):
            self._refit_menu.entryconfigure(i, label=label + (self.sym_check if i == idx else ""))

    def checkmark_criterion_options(self, idx: int):
        if idx < 0 or idx > len(self._criterion_labels):
            return
        for i, label in enumerate(self._criterion_labels):
            self._criterion_menu.entryconfigure(
                i, label=label + (self.sym_check if i == idx else "")
            )

    def exist(self):
        self._gui.mainloop()

    def shutdown(self):
        self._gui.destroy()

    def restart_command(self):
        if sys.platform == "darwin":
            return
        self.save_defaults()
        self.shutdown()

        new_frontend = Frontend()
        new_frontend.exist()


def sup(s: int):
    subs_dict = {
        "0": "\U00002070",
        "1": "\U000000B9",
        "2": "\U000000B2",
        "3": "\U000000B3",
        "4": "\U00002074",
        "5": "\U00002075",
        "6": "\U00002076",
        "7": "\U00002077",
        "8": "\U00002078",
        "9": "\U00002079",
    }
    s_str = str(s)
    ret_str = ""
    for char in s_str:
        ret_str += subs_dict[char]
    return ret_str


def sub(s: int):
    subs_dict = {
        "0": "\U00002080",
        "1": "\U00002081",
        "2": "\U00002082",
        "3": "\U00002083",
        "4": "\U00002084",
        "5": "\U00002085",
        "6": "\U00002086",
        "7": "\U00002087",
        "8": "\U00002088",
        "9": "\U00002089",
    }
    s_str = str(s)
    ret_str = ""
    for char in s_str:
        ret_str += subs_dict[char]
    return ret_str


def hexx(vec) -> str:
    if isinstance(vec, str):
        return vec
    hex_str = "#"
    for c255 in vec:
        to_add = f"{int(c255):x}"
        hex_str += to_add if len(to_add) == 2 else f"0{to_add}"
    return hex_str
