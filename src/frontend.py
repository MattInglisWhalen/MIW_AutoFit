# default libraries
import math
from dataclasses import field
from math import floor

# external libraries
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import ImageTk, Image
import os as os
import re as regex

# internal classes
from autofit.src.composite_function import CompositeFunction
from autofit.src.data_handler import DataHandler
from autofit.src.optimizer import Optimizer

class Frontend:

    def __init__(self):

        # UX
        self._new_user_stage = 1  # uses prime factors to notate which actions the user has taken

        # UI
        self._gui = tk.Tk()
        self._os_width = self._gui.winfo_screenwidth()
        self._os_height = self._gui.winfo_screenheight()

        # rendering
        self._curr_image_num = -1
        self._image_path = None
        self._image = None
        self._image_frame = None
        self._normalized_histogram_flags = []
        self._showing_fit_image = False  # conjugate to showing data-only image
        self._showing_fit_all_image = False

        # file handling
        self._filepaths = []
        self._data_handlers = []

        # text input
        self._popup_window = None
        self._excel_x_range = None
        self._excel_y_range = None
        self._excel_sigmax_range = None
        self._excel_sigmay_range = None

        # messaging
        self._num_messages_ever = 0
        self._num_messages = 0
        self._full_flag = 0

        # backend connections
        self._optimizer = None   # Optimizer
        self._model_name_tkvar = None  # tk.StringVar
        self._which5_name_tkvar = None  # tk.StringVar
        self._current_model : CompositeFunction = None
        self._current_args = None
        self._current_uncs = None
        self._curr_best_red_chi_sqr = 1e5
        self._checkbox_names_list = ["cos(x)", "sin(x)", "exp(x)", "log(x)",
                                     "1/x", "x\U000000B2", "x\U000000B3", "x\U00002074", "custom"]
        self._use_func_dict_name_tkVar = {}  # for checkboxes
        for name in self._checkbox_names_list :
            self._use_func_dict_name_tkVar[name] = tk.BooleanVar(value=False)
        self._max_functions_tkInt = tk.IntVar(value=4)
        self._brute_forcing = tk.BooleanVar(value=False)

        # defaults config
        self._default_fit_type = None
        self._default_excel_x_range = None
        self._default_excel_y_range = None
        self._default_excel_sigmax_range = None
        self._default_excel_sigmay_range = None
        self._default_load_file_loc = None
        # checkboxes default
        self.load_defaults()
        self.print_defaults()

        # load in splash screen
        self.load_splash_screen()

    def load_defaults(self):
        with open(f"{self.get_package_path()}/frontend.cfg") as file :
            for line in file :
                if "#FIT_TYPE" in line :
                    arg = regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "linear"
                    self._default_fit_type = arg
                elif "#EXCEL_RANGE_X" in line :
                    arg =  regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = "A3:A18"
                    self._default_excel_x_range = arg
                elif "#EXCEL_RANGE_Y" in line:
                    arg =  regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = ""
                    self._default_excel_y_range = arg
                elif "#EXCEL_RANGE_SIGMA_X" in line:
                    arg =  regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = ""
                    self._default_excel_sigmax_range = arg
                elif "#EXCEL_RANGE_SIGMA_Y" in line:
                    arg =  regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = ""
                    self._default_excel_sigmay_range = arg
                elif "#LOAD_FILE_LOC" in line:
                    arg =  regex.split(" ", line.rstrip("\n \t"))[-1]
                    if arg == "" or arg[0] == "#":
                        arg = f"{self.get_package_path()}/data"
                    self._default_load_file_loc = arg

    def save_defaults(self):
        with open(f"{self.get_package_path()}/frontend.cfg",'w') as file :
            file.write(f"#FIT_TYPE {self._default_fit_type}\n")
            file.write(f"#EXCEL_RANGE_X {self._default_excel_x_range}\n")
            file.write(f"#EXCEL_RANGE_Y {self._default_excel_y_range}\n")
            file.write(f"#EXCEL_RANGE_SIGMA_X {self._default_excel_sigmax_range}\n")
            file.write(f"#EXCEL_RANGE_SIGMA_Y {self._default_excel_sigmay_range}\n")
            file.write(f"#LOAD_FILE_LOC {self._default_load_file_loc}\n")

    def print_defaults(self):
        print(f">{self._default_fit_type}<")
        print(f">{self._default_excel_x_range}<")
        print(f">{self._default_excel_y_range}<")
        print(f">{self._default_excel_sigmax_range}<")
        print(f">{self._default_excel_sigmay_range}<")
        print(f">{self._default_load_file_loc}<")

    # create left, right, and middle panels
    def load_splash_screen(self):

        gui = self._gui

        # window size and title
        gui.geometry(f"{round(self._os_width*5/6)}x{round(self._os_height*5/6)}")
        gui.rowconfigure(0, minsize=800, weight=1)

        # icon image and window title
        loc = Frontend.get_package_path()
        gui.iconbitmap(f"{loc}/icon.ico")
        gui.title("AutoFit")

        # left panel -- menu buttons
        self.create_left_panel()
        self.create_load_data_button()

        # middle panel -- data visualization and fit options
        self.create_middle_panel()
        self.load_splash_image()

        # right panel -- text output
        self.create_right_panel()


    """
    
    Left Panel
    
    """

    ##
    #
    # Frames
    #
    ##

    def create_left_panel(self):
        left_panel_frame = tk.Frame(master=self._gui, relief=tk.RAISED, bg='white')
        left_panel_frame.grid(row=0, column=0, sticky='ns')

    ##
    #
    # Buttons
    #
    ##

    def create_load_data_button(self):
        load_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Load Data",
            command = self.load_data_command
        )
        load_data_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    def create_fit_button(self):
        fit_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Fit Data",
            command = self.fit_data_command
        )
        fit_data_button.grid(row=1, column=0, sticky="ew", padx=5)

    def create_fit_all_button(self):
        load_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Fit All",
            command = self.fit_all_command
        )
        load_data_button.grid(row=2, column=0, sticky="ew", padx=5)

    ##
    #
    # Button commands
    #
    ##

    def load_data_command(self):

        new_filepaths = list( fd.askopenfilenames(initialdir=self._default_load_file_loc, title="Select a file to fit",
                                                  filetypes=(("All Files", "*.*"),
                                                             ("Comma-Separated Files", "*.csv *.txt"),
                                                             ("Spreadsheets", "*.xls *.xlsx *.ods"))
                                                 )
                            )
        # trim duplicates
        for path in new_filepaths[:] :
            if path in self._filepaths :
                shortpath = regex.split( f"/", path )[-1]
                print(f"{shortpath} already loaded")
                new_filepaths.remove(path)
        for path in new_filepaths :
            if path[-4:] in [".xls","xlsx",".ods"] and self._new_user_stage % 23 != 0 :
                self.dialog_box_get_excel_data_ranges()
                if self._excel_x_range is None :
                    # the user didn't actually want to load that file
                    continue
                self._new_user_stage *= 23
                sheet_names = pd.ExcelFile(path).sheet_names
            self._default_load_file_loc = '/'.join( regex.split( f"/", path )[:-1] )
            self._filepaths.append(path)
            self._normalized_histogram_flags.append(False)

        if self._new_user_stage % 2 != 0 :
            self.create_fit_button()
            self._new_user_stage *= 2

        if len(new_filepaths) > 0 :
            self._curr_image_num = len(self._data_handlers)
            self.load_new_data(new_filepaths)
            if self._showing_fit_image :
                self._optimizer.set_data_to(self.data_handler.data)
                pars, uncs = self._optimizer.parameters_and_uncertainties_from_fitting(self._current_model)
                self._current_args = pars
                self._current_uncs = uncs
                shortpath = regex.split("/", self._filepaths[self._curr_image_num])[-1]
                self.add_message(f"\n \n> For {shortpath} \n")
                self.print_results_to_console()
                self.save_show_fit_image()
                # TODO: load new file after already obtained a fit -- the fit all button goes away when it shouldn't
            else:
                self.show_current_data()
            if self._new_user_stage % 3 != 0 :
                self.create_inspect_button()
                self._new_user_stage *= 3
            print(f"Loaded {len(new_filepaths)} files.")

        if len(self._filepaths) > 1 :
            if self._new_user_stage % 5 != 0:
                self.create_left_right_buttons()
                self._new_user_stage *= 5
            self.update_data_select()

        self.update_logx_relief()
        self.update_logy_relief()
        self.save_defaults()

    def dialog_box_get_excel_data_ranges(self):

        dialog_box = tk.Toplevel()
        dialog_box.geometry(f"{round(self._os_width/4)}x{round(self._os_height/4)}")
        dialog_box.title("Spreadsheet Input Options")
        dialog_box.iconbitmap(f"{self.get_package_path()}/icon.ico")

        x_label = tk.Label(master=dialog_box, text="Cells for x values: ")
        x_label.grid(row=0,column=0)
        x_data = tk.Entry(master=dialog_box)
        x_data.insert(0,self._default_excel_x_range)
        x_data.grid(row=0,column=1, sticky='w')

        y_label = tk.Label(master=dialog_box, text="Cells for y values: ")
        y_label.grid(row=1,column=0)
        y_data = tk.Entry(master=dialog_box)
        y_data.insert(0,self._default_excel_y_range)
        y_data.grid(row=1,column=1, sticky='w')

        sigmax_label = tk.Label(master=dialog_box, text="Cells for x uncertainties: ")
        sigmax_label.grid(row=2,column=0)
        sigmax_data = tk.Entry(master=dialog_box)
        sigmax_data.insert(0,self._default_excel_sigmax_range)
        sigmax_data.grid(row=2,column=1, sticky='w')

        sigmay_label = tk.Label(master=dialog_box, text="Cells for y uncertainties: ")
        sigmay_label.grid(row=3,column=0)
        sigmay_data = tk.Entry(master=dialog_box)
        sigmay_data.insert(0,self._default_excel_sigmay_range)
        sigmay_data.grid(row=3,column=1, sticky='w')

        close_dialog_button = tk.Button(
            master = dialog_box,
            text="OK",
            command=self.close_dialog_box_command
        )
        close_dialog_button.grid(row=0,column=10,sticky='ns')
        dialog_box.bind('<Return>', self.close_dialog_box_command)
        dialog_box.focus_force()

        self._popup_window = dialog_box
        self._gui.wait_window(dialog_box)


    def close_dialog_box_command(self, bind_command=None):

        if self._popup_window is None :
            print("Window already closed")
        self._excel_x_range = self._popup_window.children['!entry'].get()
        self._excel_y_range = self._popup_window.children['!entry2'].get()
        self._excel_sigmax_range = self._popup_window.children['!entry3'].get()
        self._excel_sigmay_range = self._popup_window.children['!entry4'].get()

        self._default_excel_x_range = self._excel_x_range
        self._default_excel_y_range = self._excel_y_range
        self._default_excel_sigmax_range = self._excel_sigmax_range
        self._default_excel_sigmay_range = self._excel_sigmay_range

        self.save_defaults()
        self._popup_window.destroy()


    ##
    #
    # Helper functions
    #
    ##

    def load_new_data(self, new_filepaths_lists):
        for path in new_filepaths_lists :
            if path[-4:] in [".xls","xlsx",".ods"] :
                for idx, sheet_name in enumerate(pd.ExcelFile(path).sheet_names):
                    self._data_handlers.append(DataHandler(filepath=path))
                    self._data_handlers[-1].set_excel_args(x_range_str=self._excel_x_range,
                                                           y_range_str=self._excel_y_range,
                                                           x_error_str=self._excel_sigmax_range,
                                                           y_error_str=self._excel_sigmay_range)
                    self._data_handlers[-1].set_excel_sheet_name( sheet_name )
                    break  # stand-in for additional excel options panel

            else:
                # only add one data handler
                self._data_handlers.append(DataHandler(filepath=path))


    def reload_all_data(self):
        self._data_handlers = []
        for path in self._filepaths :
            self._data_handlers.append(DataHandler(filepath=path))

    def show_data(self, file_num=0):

        # mod_file_num = file_num % len(self._data_handlers)

        new_image_path = f"{Frontend.get_package_path()}/plots/front_end_current_plot.png"
        # create a scatter plot of the first file

        x_points = self.data_handler.unlogged_x_data
        y_points =  self.data_handler.unlogged_y_data
        sigma_x_points = self.data_handler.unlogged_sigmax_data
        sigma_y_points = self.data_handler.unlogged_sigmay_data

        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
        plt.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        plt.xlabel(self.data_handler.x_label)
        plt.ylabel(self.data_handler.y_label)
        axes = plt.gca()
        if axes.get_xlim()[0] > 0:
            axes.set_xlim([0, axes.get_xlim()[1]])
        elif axes.get_xlim()[1] < 0:
            axes.set_xlim([axes.get_xlim()[0], 0])
        if axes.get_ylim()[0] > 0:
            axes.set_ylim([0, axes.get_ylim()[1]])
        elif axes.get_ylim()[1] < 0:
            axes.set_ylim([axes.get_ylim()[0], 0])

        # print(f"Log flags : {self.data_handler.logx_flag} {self.data_handler.logy_flag}")
        if self.data_handler.logx_flag:
            print("Setting log xscale in show_data")
            log_min, log_max = math.log(min(x_points)), math.log(max(x_points))
            print(log_min, log_max, math.exp(log_min), math.exp(log_max))
            axes.set_xlim([math.exp(log_min - (log_max - log_min) / 10), math.exp(log_max + (log_max - log_min) / 10)])
            axes.set(xscale="log")
            axes.spines['right'].set_visible(False)
        else:
            axes.set(xscale="linear")
            axes.spines['left'].set_position(('data', 0.))
            axes.spines['right'].set_position(('data', 0.))
            axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
        if self.data_handler.logy_flag:
            print("Setting log xscale in show_data")
            axes.set(yscale="log")
            log_min, log_max = math.log(min(y_points)), math.log(max(y_points))
            axes.set_ylim([math.exp(log_min - (log_max - log_min) / 10), math.exp(log_max + (log_max - log_min) / 10)])
            axes.spines['top'].set_visible(False)
        else:
            axes.set(yscale="linear")
            axes.spines['top'].set_position(('data', 0.))
            axes.spines['bottom'].set_position(('data', 0.))
            axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
        axes.set_facecolor((112 / 255, 146 / 255, 190 / 255))

        min_X, max_X = min(x_points), max(x_points)
        min_Y, max_Y = min(y_points), max(y_points)
        #  proportion between xmin and xmax where the zero lies
        # x(tx) = xmin + (xmax - xmin)*tx with 0<tx<1 so
        tx = max(0,-min_X / (max_X - min_X))
        ty = max(0,-min_Y / (max_Y - min_Y))
        offset_X, offset_Y = -0.1, 0.0  # how much of the screen is taken by the x and y spines

        axes.xaxis.set_label_coords( 1.050,offset_Y+ty)
        axes.yaxis.set_label_coords( offset_X+tx,+0.750)

        plt.tight_layout()
        plt.savefig(new_image_path)

        # replace the splash graphic with the plot
        self._image_path = new_image_path
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure( image = self._image )

        # if we're showing the image, we want the optimizer to be working with this data
        if self._showing_fit_image :
            data = self.data_handler.data
            self._optimizer.set_data_to(data)

        # add logx and logy to plot options frame
        if self._new_user_stage % 13 != 0:
            self.create_logx_button()
            self.create_logy_button()
            self._new_user_stage *= 13

        # if it's a histogram, add an option to normalize the data to the plot options frame
        if self.data_handler.histogram_flag :
            if self._new_user_stage % 17 != 0 :
                self.create_normalize_button()
                self._new_user_stage *= 17
            else:
                # button already exists, but might be hidden
                if self.data_handler.logx_flag or self.data_handler.logy_flag :
                    pass
                else:
                    self.show_normalize_button()
                # for a good reason though!
        else:
            self.hide_normalize_button()



    def fit_data_command(self):

        # add buttons to adjust fit options
        if self._new_user_stage % 7 != 0 :
            self._new_user_stage *= 7
            self.create_function_dropdown()

        # Find the fit for the currently displayed data
        data = self.data_handler.data
        self._optimizer = Optimizer(data=data,
                                    use_functions_dict = self.use_functions_dict(),
                                    max_functions=self.max_functions() )
        plot_model = None
        if self._model_name_tkvar.get() == "Linear" :
            print("Fitting to linear model")
            plot_model = CompositeFunction.built_in("Linear")
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model)
        elif self._model_name_tkvar.get() == "Gaussian" and self.normalized_histogram_flag:
            print("Fitting to Normal distribution")
            plot_model = CompositeFunction.built_in("Normal")
            initial_guess = self._optimizer.find_initial_guess_scaling(plot_model)
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model,initial_guess=initial_guess)
        elif self._model_name_tkvar.get() == "Gaussian" :
            print("Fitting to Gaussian model")
            plot_model = CompositeFunction.built_in("Gaussian")
            initial_guess = self._optimizer.find_initial_guess_scaling(plot_model)
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model,initial_guess=initial_guess)
        elif self._model_name_tkvar.get() == "Sigmoid" :
            print("Fitting to Sigmoid model")
            plot_model = CompositeFunction.built_in("Sigmoid")
            initial_guess = self._optimizer.find_initial_guess_scaling(plot_model)
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model,initial_guess=initial_guess)
        elif self._model_name_tkvar.get() == "Procedural":
            print("Fitting to procedural model")
            # find fit button should now be find model
            self._optimizer.find_best_model_for_dataset()
            plot_model = self._optimizer.best_model
        elif self._model_name_tkvar.get() == "Brute-Force":
            print("Brute forcing a procedural model")
            # find fit button should now be find model
            self.brute_forcing = True
            for name in self._checkbox_names_list:
                self._use_func_dict_name_tkVar[name].set(value=True)
                # pass
            self._optimizer.load_non_defaults_from(self.use_functions_dict())
            self._optimizer.async_find_best_model_for_dataset(start=True)
            plot_model = self._optimizer.best_model
        else:
            print(f"Invalid model name {self._model_name_tkvar.get()}")
            pass
        self._current_model = plot_model
        self._current_args = self._optimizer.parameters
        self._current_uncs = self._optimizer.uncertainties
        self.save_show_fit_image()

        # add fit all button if there's more than one file
        if self._new_user_stage % 11 != 0 and len(self._data_handlers) > 1 :
            self.create_fit_all_button()
            self._new_user_stage *= 11

        # print out the parameters on the right
        shortpath = regex.split("/", self._filepaths[self._curr_image_num])[-1]
        self.add_message(f"\n \n> For {shortpath} \n")
        self.print_results_to_console()
        self._default_fit_type = self._model_name_tkvar.get()
        self.save_defaults()

        if self._new_user_stage % 29 == 0 and self._model_name_tkvar.get() == "Procedural":
            self.update_top5_dropdown()

        # add a dropdown list for procedural-type fits
        if self._model_name_tkvar.get() in ["Procedural","Brute-Force"] and self._new_user_stage % 29 != 0:
            self.create_top5_dropdown()
            self._new_user_stage *= 29

        # add status updates for brute-force fits
        if self._model_name_tkvar.get() == "Procedural" and self._new_user_stage % 31 != 0 :
            self.create_default_checkboxes()
            self.create_depth_up_down_buttons()
            self._new_user_stage *= 31

        if self._model_name_tkvar.get() == "Brute-Force" and self._new_user_stage % 37 != 0 :
            self.create_pause_button()
            self._new_user_stage *= 37

        if self.brute_forcing :
            self.begin_brute_loop()

    def begin_brute_loop(self):
        self._gui.update_idletasks()
        self._gui.after_idle(self.maintain_brute_loop)

    def maintain_brute_loop(self):
        if self.brute_forcing :
            status = self._optimizer.async_find_best_model_for_dataset()
            if status == "Done" :
                self.brute_forcing = False
                print("End of brute-forcing reached")
            self._current_model = self._optimizer.best_model
            self._current_args = self._optimizer.parameters
            self._current_uncs = self._optimizer.uncertainties
            self.update_top5_dropdown()
            if self._curr_best_red_chi_sqr != self._optimizer.top5_rx_sqrs[0] :
                self.save_show_fit_image()
                self._curr_best_red_chi_sqr = self._optimizer.top5_rx_sqrs[0]
            self._gui.after(1, self.maintain_brute_loop)
        self.update_pause_button()

    def print_results_to_console(self):
        print_string = ""
        if self._model_name_tkvar.get() == "Linear" :
            if self.data_handler.logy_flag :
                print_string += f"\n>  Linear fit is LY ="
            else :
                print_string += f"\n>  Linear fit is y ="
            if self.data_handler.logx_flag :
                print_string += f" m LX + b with\n"
            else :
                print_string += f" m x + b with\n"
            m, sigmam = self._optimizer.parameters[0], self._optimizer.uncertainties[0]
            b = self._optimizer.parameters[0]*self._optimizer.parameters[1]
            sigmab = math.sqrt( self._optimizer.uncertainties[0]**2 * self._optimizer.parameters[1]**2 +
                                self._optimizer.uncertainties[1]**2 * self._optimizer.parameters[0]**2   )
            print_string += f"   m = {m:+.2E}  \u00B1  {sigmam:.2E}\n"
            print_string += f"   b = {b:+.2E}  \u00B1  {sigmab:.2E}\n"
            print_string += f"Goodness of fit: R\U000000B2 = {self._optimizer.r_squared(self._optimizer.best_model):.2F}\n"
        elif self._model_name_tkvar.get() == "Gaussian" and self.normalized_histogram_flag:
            if self.data_handler.logy_flag :
                print_string += f"\n>  Normal fit is LY ="
            else :
                print_string += f"\n>  Normal fit is y ="
            if self.data_handler.logx_flag :
                print_string += f" 1/\u221A(2\u03C0\u03C3\U000000B2) exp[-(LX-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
            else :
                print_string += f" 1/\u221A(2\u03C0\u03C3\U000000B2) exp[-(x-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
            mu, sigmamu = -self._optimizer.parameters[1], self._optimizer.uncertainties[1]
            sigma = math.sqrt( 1 / (2*math.pi*self._optimizer.parameters[0]**2) )
            sigmasigma = 1/(2*math.pi*self._optimizer.parameters[0]**2) * self._optimizer.uncertainties[0]
            print_string += f"   \u03BC = {mu:+.2E}  \u00B1  {sigmamu:.2E}\n"
            print_string += f"   \u03C3 =  {sigma:.2E}  \u00B1  {sigmasigma:.2E}\n"
        elif self._model_name_tkvar.get() == "Gaussian" :
            if self.data_handler.logy_flag :
                print_string += f"\n>  Gaussian fit is LY ="
            else :
                print_string += f"\n>  Gaussian fit is y ="
            if self.data_handler.logx_flag :
                print_string += f" A exp[-(LX-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
            else :
                print_string += f" A exp[-(x-\u03BC)\U000000B2/2\u03C3\U000000B2] with\n"
            A, sigmaA = self._optimizer.parameters[0], self._optimizer.uncertainties[0]
            mu, sigmamu = -self._optimizer.parameters[2], self._optimizer.uncertainties[2]
            sigma = math.sqrt( -1 / (2*self._optimizer.parameters[1]) )
            sigmasigma = math.sqrt( 1/(4*self._optimizer.parameters[1]**2 * sigma) ) * self._optimizer.uncertainties[1]
            print_string += f"   A = {A:+.2E}  \u00B1  {sigmaA:.2E}\n"
            print_string += f"   \u03BC = {mu:+.2E}  \u00B1  {sigmamu:.2E}\n"
            print_string += f"   \u03C3 =  {sigma:.2E}  \u00B1  {sigmasigma:.2E}\n"
        elif self._model_name_tkvar.get() == "Sigmoid" :
            # print_string += f"  Sigmoid fit is y = F + H/(1 + exp[-(x-x0)/w] )\n"
            if self.data_handler.logy_flag :
                print_string += f"\n>  Sigmoid fit is LY ="
            else :
                print_string += f"\n>  Sigmoid fit is y ="
            if self.data_handler.logx_flag :
                print_string += f" F + H/(1 + exp[-(LX-x0)/w] ) with\n"
            else :
                print_string += f" F + H/(1 + exp[-(x-x0)/w] ) with\n"
            F, sigmaF = self._optimizer.parameters[0], self._optimizer.uncertainties[0]
            H = self._optimizer.parameters[0]*self._optimizer.parameters[1]
            sigmaH = math.sqrt( self._optimizer.uncertainties[0]**2 * self._optimizer.parameters[1]**2 +
                                self._optimizer.uncertainties[1]**2 * self._optimizer.parameters[0]**2   )
            w = -1/self._optimizer.parameters[3]
            sigmaW = self._optimizer.uncertainties[3]/self._optimizer.parameters[3]**2
            try :
                x0 = w*math.log(self._optimizer.parameters[2])
            except ValueError :
                print(f"Can't take the log of {self._optimizer.parameters[2]}")
                raise ValueError
            sigmax0 = math.sqrt( sigmaW**2 * (x0/w)**2 +
                                 w**2 * self._optimizer.uncertainties[2]**2 / self._optimizer.parameters[2]**2  )

            print_string += f"   F  = {F:+.2E}  \u00B1  {sigmaF:.2E}\n"
            print_string += f"   H  =  {H:.2E}  \u00B1  {sigmaH:.2E}\n"
            print_string += f"   w  =  {w:.2E}  \u00B1  {sigmaW:.2E}\n"
            print_string += f"   x0 = {x0:+.2E}  \u00B1  {sigmax0:.2E}\n"
        elif self._model_name_tkvar.get() == "Procedural":
            if self.data_handler.logy_flag :
                print_string += f"\n> Selected model is LY = {self._current_model.name}"
            else :
                print_string += f"\n> Selected model is y = {self._current_model.name}"
            if self.data_handler.logx_flag :
                print_string += f"(LX) w/ {self._current_model.dof} dof and where\n"
            else :
                print_string += f"(x) w/ {self._current_model.dof} dof and where\n"
            for idx, (par, unc) in enumerate(zip(self._current_args, self._current_uncs)):
                print_string += f"  c{idx} =  {par:+.2E}  \u00B1  {unc:.2E}\n"
            print_string += "\n> As a tree, this is \n"
            print_string += self._current_model.tree_as_string_with_args()
        elif self._model_name_tkvar.get() == "Brute-Force":
            if self.data_handler.logy_flag :
                print_string += f"\n> Model is LY = {self._current_model.name}"
            else :
                print_string += f"\n> Model is y = {self._current_model.name}"
            if self.data_handler.logx_flag :
                print_string += f"(LX) w/ {self._current_model.dof} dof and where\n"
            else :
                print_string += f"(x) w/ {self._current_model.dof} dof and where\n"
            for idx, (par, unc) in enumerate(zip(self._current_args, self._current_uncs)):
                print_string += f"  c{idx} =  {par:+.2E}  \u00B1  {unc:.2E}\n"
            print_string += "\n> As a tree, this is \n"
            print_string += self._current_model.tree_as_string_with_args()
        else:
            pass
        if self.data_handler.logy_flag and self.data_handler.logx_flag:
            print_string += f"Keep in mind that LY = log(y/{self.data_handler.Y0:.2E}) and LX = log(x/{self.data_handler.X0:.2E})\n"
        elif self.data_handler.logy_flag :
            print_string += f"Keep in mind that LY = log(y/{self.data_handler.Y0:.2E})\n"
        elif self.data_handler.logx_flag :
            print_string += f"Keep in mind that LX = log(x/{self.data_handler.X0:.2E})\n"
        self.add_message(print_string)

    def fit_all_command(self):

        self.add_message("\n \n> Fitting all datasets\n")

        # if self._optimizer is None :
        #     # have to first find an optimal model
        #     self.add_message("> Finding optimal model for current dataset\n")
        #     self.fit_data_command()

        # need to log all datasets if the current one is logged, and unlog if they ARE logged
        for handler in self._data_handlers :
            if handler == self.data_handler :
                continue
            if self.data_handler.logx_flag :
                if handler.logx_flag :
                    # unlog then relog
                    handler.logx_flag = False
                handler.X0 = -self.data_handler.X0  # links the two X0 values
                handler.logx_flag = True
            elif not self.data_handler.logx_flag and handler.logx_flag :
                handler.logx_flag = False

            if self.data_handler.logy_flag :
                if handler.logy_flag :
                    # unlog then relog
                    handler.logy_flag = False
                handler.Y0 = -self.data_handler.Y0  # links the two Y0 values
                handler.logy_flag = True
            elif not self.data_handler.logy_flag and handler.logy_flag :
                handler.logy_flag = False
            # TODO: test that this works
            # it's a little complicated because the X0 and Y0 values are different


        # need to normalize all datasets if the current one is normalized
        if any([handler.normalized for handler in self._data_handlers]) :
            self.data_handler.normalize_histogram_data()
        for handler in self._data_handlers :
            if self.data_handler.normalized and not handler.normalized :
                handler.normalize_histogram_data()

        # fit every loaded dataset with the current model and return the average parameters
        list_of_args = []
        list_of_uncertainties = []
        for handler in self._data_handlers :
            data = handler.data
            self._optimizer.set_data_to(data)
            # does the following line actually use the chosen model?
            pars, uncertainties = self._optimizer.parameters_and_uncertainties_from_fitting(model=self._current_model,
                                                                                            initial_guess=self._optimizer.best_model.get_args())
            list_of_args.append(pars)
            list_of_uncertainties.append(uncertainties)
            print(f"Fit pars = {pars}")
            # self._current_model.set_args(*pars)
            # self.save_show_fit_image(model=self._current_model)

        self.add_message("> Average parameters from fitting all datasets:\n")
        means = []
        uncs = []
        for idx, _ in enumerate(list_of_args[0]) :
            N = len(list_of_args)
            sum_args = 0
            for par_list in list_of_args :
                sum_args += par_list[idx]
            mean = sum_args/N

            sum_uncertainty_sqr = 0
            sum_variance = 0
            for par_list, unc_list in zip(list_of_args,list_of_uncertainties) :
                sum_uncertainty_sqr += unc_list[idx]**2 / N
                sum_variance += (par_list[idx]-mean)**2 / (N-1) if N > 1 else 0

            ratio = sum_variance / (sum_variance + sum_uncertainty_sqr )
            effective_variance = ratio * sum_variance + (1-ratio) * sum_uncertainty_sqr

            means.append(mean)
            uncs.append(math.sqrt(effective_variance))

        self._current_args = means
        self._current_uncs = uncs
        self._current_model.set_args(*means)

        self.save_show_fit_all(model=self._current_model, args_list = list_of_args)

        # TODO: figure out what to do with the left/right arrows and the numbers

        self.print_results_to_console()
        self.update_data_select()


    """

    Middle Panel

    """

    def create_middle_panel(self):
        self._gui.columnconfigure(1, minsize=720)  # image panel
        middle_panel_frame = tk.Frame(master=self._gui, relief=tk.RIDGE)
        middle_panel_frame.grid(row=0, column=1, sticky='news')
        self.create_image_frame()
        self.create_data_perusal_frame()
        self.create_fit_options_frame()
        self.create_plot_options_frame()
        self.create_depth_frame()

    ##
    #
    # Frames
    #
    ##

    def create_image_frame(self):  # !frame : image only
        self._gui.children['!frame2'].columnconfigure(1,minsize=50)
        image_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        image_frame.grid(row=0, column=0, sticky='w')
        self.load_splash_image()

    def create_data_perusal_frame(self):  # !frame2 : inspect, left<>right buttons
        data_perusal_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        data_perusal_frame.grid(row=1, column=0, sticky='w')

    def create_fit_options_frame(self):  # !frame3 : fit type, procedural top5, procedural checkboxes
        fit_options_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        fit_options_frame.grid(row = 3, column=0, sticky='w')

    def create_plot_options_frame(self):  # !frame4 : logx, logy, normalize
        plot_options_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        plot_options_frame.grid(row = 0, column=1, sticky='ns')

    def create_depth_frame(self):  # !frame5 : depth of procedural fits
        depth_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        depth_frame.grid(row = 4, column=0, sticky = 'w')


    ##
    #
    # Buttons
    #
    ##

    def create_inspect_button(self):

        # TODO: also make a save figure button

        data_perusal_button = tk.Button(
            master = self._gui.children['!frame2'].children['!frame2'],
            text = "Inspect",
            command = self.inspect_command
        )
        data_perusal_button.grid(row=0, column=0, padx=5, pady=5)

    def create_left_right_buttons(self):

        left_button = tk.Button( master = self._gui.children['!frame2'].children['!frame2'],
                                 text = "\U0001F844",
                                 command = self.image_left_command
                               )
        count_text = tk.Label(
            master=self._gui.children['!frame2'].children['!frame2'],
            text = f"{self._curr_image_num % len(self._data_handlers) + 1}/{len(self._data_handlers)}"
        )
        right_button = tk.Button( master = self._gui.children['!frame2'].children['!frame2'],
                                  text = "\U0001F846",
                                  command = self.image_right_command
                                )
        left_button.grid(row=0, column=1, padx=5, pady=5)
        count_text.grid(row=0, column=2)
        right_button.grid(row=0, column=3, padx=5, pady=5)

    def create_show_residuals_button(self):
        show_residuals_button = tk.Button(
            master = self._gui.children['!frame2'].children['!frame2'],
            text = "Show Residuals",
            command = self.show_residuals_command
        )
        show_residuals_button.grid(row=0, column=4, padx=5, pady=5, sticky = 'e')

    def create_function_dropdown(self):

        # black line above frame 3
        self._gui.children['!frame2'].rowconfigure(2, minsize=1)
        black_line_as_frame = tk.Frame(
            master=self._gui.children['!frame2'],
            bg = 'black'
        )
        black_line_as_frame.grid(row = 2, column=0, sticky='ew')

        func_list = ["Linear", "Gaussian", "Sigmoid", "Procedural", "Brute-Force"]

        self._model_name_tkvar = tk.StringVar(self._gui.children['!frame2'].children['!frame3'])
        self._model_name_tkvar.set(self._default_fit_type)

        function_dropdown = tk.OptionMenu(
            self._gui.children['!frame2'].children['!frame3'],
            self._model_name_tkvar,
            *func_list
        )
        function_dropdown.configure(width=9)
        function_dropdown.grid(row=0, column=0)

        self._model_name_tkvar.trace('w', self.function_dropdown_trace)

    def create_top5_dropdown(self):

        # top 5 fits quick list

        # print("Creating", self._gui.children['!frame2'].children['!frame3'].children)

        top5_list = [ f"{rx_sqr:.2F}: {name}" for rx_sqr, name
                      in zip(self._optimizer.top5_rx_sqrs, self._optimizer.top5_names)]

        self._which5_name_tkvar = tk.StringVar(self._gui.children['!frame2'].children['!frame3'])
        self._which5_name_tkvar.set("Top 5")

        top5_dropdown = tk.OptionMenu(
            self._gui.children['!frame2'].children['!frame3'],
            self._which5_name_tkvar,
            *top5_list
        )
        top5_dropdown.configure(width=45)
        top5_dropdown.grid(row=0, column=1)

        self._which5_name_tkvar.trace('w', self.which5_dropdown_trace)

    def update_top5_dropdown(self):
        # print("Updating", self._gui.children['!frame2'].children['!frame3'].children)
        top5_dropdown : tk.OptionMenu = self._gui.children['!frame2'].children['!frame3'].children['!optionmenu2']

        top5_dropdown['menu'].delete(0,tk.END)
        top5_list = [f"{rx_sqr:.2F}: {name}" for rx_sqr, name
                     in zip(self._optimizer.top5_rx_sqrs, self._optimizer.top5_names)]
        for label in top5_list :
            top5_dropdown['menu'].add_command(label=label, command=tk._setit(self._which5_name_tkvar, label))


    def hide_top5_dropdown(self):
        # print("Hiding",self._gui.children['!frame2'].children['!frame3'].children)
        if self._new_user_stage % 29 != 0 :
            # the dropdown hasn't been created yet, no need to hide it
            return
        top5_dropdown = self._gui.children['!frame2'].children['!frame3'].children['!optionmenu2']
        top5_dropdown.grid_forget()
    def show_top5_dropdown(self):
        # print("Showing",self._gui.children['!frame2'].children['!frame3'].children)
        top5_dropdown = self._gui.children['!frame2'].children['!frame3'].children['!optionmenu2']
        top5_dropdown.grid(row=0, column=1)

    def create_logx_button(self):
        log_x_button = tk.Button(
            master = self._gui.children['!frame2'].children['!frame4'],
            text = "Log X",
            command = self.logx_command
        )
        log_x_button.grid(row=0, column=0, padx=5, pady=(5,0), sticky = 'w')

    def create_logy_button(self):
        log_y_button = tk.Button(
            master = self._gui.children['!frame2'].children['!frame4'],
            text = "Log Y",
            command = self.logy_command
        )
        log_y_button.grid(row=1, column=0, padx=5, sticky = 'w')

    def create_normalize_button(self):
        normalize_button = tk.Button(
            master = self._gui.children['!frame2'].children['!frame4'],
            text = "Normalize",
            command = self.normalize_command
        )
        normalize_button.grid(row=2, column=0, padx=5, pady=5, sticky = 'w')

    def hide_normalize_button(self):
        if self._new_user_stage % 17 != 0 :
            # the button hasn't been created yet, no need to hide it
            return
        normalize_button = self._gui.children['!frame2'].children['!frame4'].children['!button3']
        normalize_button.grid_forget()
    def show_normalize_button(self):
        normalize_button = self._gui.children['!frame2'].children['!frame4'].children['!button3']
        normalize_button.grid(row=2, column=0, padx=5, pady=5, sticky = 'w')

    ##
    #
    # Button commands
    #
    ##

    def inspect_command(self):

        plt.show()

        # TODO: find a way to show() again without rerunning fits

        if self._showing_fit_all_image :
            self.save_show_fit_all()
        elif self._showing_fit_image :
            self.save_show_fit_image()
        else:
            self.show_data()

    def image_left_command(self):
        self._curr_image_num = (self._curr_image_num - 1) % len(self._data_handlers)
        self._showing_fit_all_image = False

        if self._showing_fit_image :
            self._optimizer.set_data_to(self.data_handler.data)
            pars, uncs = self._optimizer.parameters_and_uncertainties_from_fitting(self._current_model)
            self._current_args = pars
            self._current_uncs = uncs
            shortpath = regex.split("/", self._filepaths[self._curr_image_num])[-1]
            self.add_message(f"\n \n> For {shortpath} \n")
            self.print_results_to_console()
            self.save_show_fit_image()
        else:
            self.show_current_data()

        self.update_data_select()

        if self.data_handler.histogram_flag :
            self.show_normalize_button()
        else :
            self.hide_normalize_button()
        self.update_logx_relief()
        self.update_logy_relief()


    def image_right_command(self):
        self._curr_image_num  = (self._curr_image_num + 1) % len(self._data_handlers)
        # change pars and uncs to current
        self._showing_fit_all_image = False

        if self._showing_fit_image :
            self._optimizer.set_data_to(self.data_handler.data)
            pars, uncs = self._optimizer.parameters_and_uncertainties_from_fitting(self._current_model)
            self._current_args = pars
            self._current_uncs = uncs
            shortpath = regex.split("/", self._filepaths[self._curr_image_num])[-1]
            self.add_message(f"\n \n> For {shortpath} \n")
            self.print_results_to_console()
            self.save_show_fit_image()
        else:
            self.show_current_data()

        self.update_data_select()

        if self.data_handler.histogram_flag :
            self.show_normalize_button()
        else :
            self.hide_normalize_button()
        self.update_logx_relief()
        self.update_logy_relief()

    def show_residuals_command(self):
        # TODO: this would be cool
        pass

    def normalize_command(self):
        self.data_handler.normalize_histogram_data()
        self.normalized_histogram_flag = True
        if self._showing_fit_image :
            self.fit_data_command()
        else:
            self.show_current_data()

    def update_logx_relief(self):
        button: tk.Button = self._gui.children['!frame2'].children['!frame4'].children['!button']
        if self.data_handler.logx_flag :
            button.configure(relief=tk.SUNKEN)
            self.hide_normalize_button()
        else:
            button.configure(relief=tk.RAISED)
            if not self.data_handler.logy_flag and self.data_handler.histogram_flag :
                self.show_normalize_button()

    # TODO: if loading new file, or using left/right buttons, need to update the logx and logy buttons
    def logx_command(self):

        # flip-flop
        if self.data_handler.logx_flag :
            self.data_handler.logx_flag = False
        else:
            self.data_handler.logx_flag = True

        self.update_logx_relief()

        if self._optimizer is not None:
            self._optimizer.set_data_to(self.data_handler.data)
        if self._showing_fit_image :
            self._optimizer.parameters_and_uncertainties_from_fitting(self._current_model)
            self.save_show_fit_image()
        else:
            self.show_current_data()

        # TODO: clear the top5 models list, since the top5 stored models fit "different" data


    def update_logy_relief(self):
        button: tk.Button = self._gui.children['!frame2'].children['!frame4'].children['!button2']
        if self.data_handler.logy_flag :
            button.configure(relief=tk.SUNKEN)
            self.hide_normalize_button()
        else:
            button.configure(relief=tk.RAISED)
            if not self.data_handler.logx_flag and self.data_handler.histogram_flag :
                self.show_normalize_button()

    def logy_command(self):

        if self.data_handler.logy_flag :
            self.data_handler.logy_flag = False
        else:
            self.data_handler.logy_flag = True

        self.update_logy_relief()

        if self._optimizer is not None:
            self._optimizer.set_data_to(self.data_handler.data)
        if self._showing_fit_image:
            self._optimizer.parameters_and_uncertainties_from_fitting(self._current_model)
            self.save_show_fit_image()
        else:
            self.show_current_data()

        # TODO: clear the top5 models list, since the top5 stored models fit "different" data


    ##
    #
    # Helper functions
    #
    ##

    def load_splash_image(self):
        self._image_path = f"{Frontend.get_package_path()}/splash.png"
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame = tk.Label(master=self._gui.children['!frame2'].children['!frame'],
                                     image=self._image,
                                     relief=tk.SUNKEN)
        self._image_frame.grid(row=0, column=0)

    def update_data_select(self):
        text_label = self._gui.children['!frame2'].children['!frame2'].children['!label']
        if self._showing_fit_all_image :
            text_label.configure(text=f"-/{len(self._data_handlers)}")
        else:
            text_label.configure(
                text=f"{(self._curr_image_num % len(self._data_handlers)) + 1 }/{len(self._data_handlers)}"
            )

    def function_dropdown_trace(self,*args):
        model_choice = self._model_name_tkvar.get()

        self.fit_data_command()

        if model_choice in ["Procedural", "Brute-Force"] :
            if self._new_user_stage % 29 != 0 :
                self.create_top5_dropdown()
                self._new_user_stage *= 29
            else :
                self.show_top5_dropdown()
                self.update_top5_dropdown()
        else:
            self.hide_top5_dropdown()

        if model_choice == "Procedural" :
            if self._new_user_stage % 31 != 0 :
                self.create_default_checkboxes()
                self.create_depth_up_down_buttons()
                self._new_user_stage *= 31
            else :
                self.show_default_checkboxes()
                self.show_depth_buttons()
        else:
            self.hide_default_checkboxes()
            self.hide_depth_buttons()

        if model_choice != "Brute-Force" :
            self.brute_forcing = False
            self.hide_pause_button()

    def save_show_fit_image(self, model = None):

        if model is not None:
            plot_model = model.copy()
        else:
            plot_model = self._current_model

        handler = self.data_handler

        x_points = handler.unlogged_x_data
        y_points = handler.unlogged_y_data
        sigma_x_points = handler.unlogged_sigmax_data
        sigma_y_points = handler.unlogged_sigmay_data

        smooth_x_for_fit = np.linspace( x_points[0], x_points[-1], 4*len(x_points))
        if handler.logx_flag and handler.logy_flag :
            fit_vals = [ plot_model.eval_at(xi, X0 = self.data_handler.X0, Y0 = self.data_handler.Y0)
                         for xi in smooth_x_for_fit ]
        elif handler.logx_flag :
            fit_vals = [ plot_model.eval_at(xi, X0 = self.data_handler.X0) for xi in smooth_x_for_fit ]
        elif handler.logy_flag :
            fit_vals = [ plot_model.eval_at(xi, Y0 = self.data_handler.Y0) for xi in smooth_x_for_fit ]
        else:
            fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]

        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor((112 / 255, 146 / 255, 190 / 255))
        plt.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        plt.plot(smooth_x_for_fit, fit_vals, '-', color='r')
        plt.xlabel(handler.x_label)
        plt.ylabel(handler.y_label)
        axes : plt.axes = plt.gca()
        if axes.get_xlim()[0] > 0:
            axes.set_xlim([0, axes.get_xlim()[1]])
        elif axes.get_xlim()[1] < 0:
            axes.set_xlim([axes.get_xlim()[0], 0])
        if axes.get_ylim()[0] > 0:
            axes.set_ylim([0, axes.get_ylim()[1]])
        elif axes.get_ylim()[1] < 0:
            axes.set_ylim([axes.get_ylim()[0], 0])

        if handler.logx_flag:
            log_min, log_max = math.log(min(x_points)), math.log(max(x_points))
            axes.set_xlim([math.exp(log_min - (log_max - log_min) / 10), math.exp(log_max + (log_max - log_min)/10)])
            axes.set(xscale="log")
            axes.spines['right'].set_visible(False)
        else:
            axes.set(xscale="linear")
            axes.spines['left'].set_position(  ('data', 0.) )
            axes.spines['right'].set_position( ('data', 0.) )
            axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
        if handler.logy_flag:
            axes.set(yscale="log")
            log_min, log_max = math.log(min( y_points )), math.log(max(y_points))
            axes.set_ylim([math.exp(log_min - (log_max-log_min)/10), math.exp(log_max + (log_max-log_min)/10)])
            axes.spines['top'].set_visible(False)
        else:
            axes.set(yscale="linear")
            axes.spines['top'].set_position(    ('data', 0.) )
            axes.spines['bottom'].set_position( ('data', 0.) )
            axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
        axes.set_facecolor((112 / 255, 146 / 255, 190 / 255))

        # print( np.array(axes.spines['top'].get_spine_transform()) )

        min_X, max_X = min(x_points), max(x_points)
        min_Y, max_Y = min(y_points), max(y_points)
        #  tx is the proportion between xmin and xmax where the zero lies
        # x(tx) = xmin + (xmax - xmin)*tx with 0<tx<1 so
        tx = max(0,-min_X / (max_X - min_X))
        ty = max(0,-min_Y / (max_Y - min_Y))
        offset_X, offset_Y = -0.1, 0.0  # how much of the screen is taken by the x and y spines

        axes.xaxis.set_label_coords( 1.050,offset_Y+ty)
        axes.yaxis.set_label_coords( offset_X+tx,+0.750)

        plt.tight_layout()
        plt.savefig(self._image_path)

        # change the view to show the fit as well
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure(image=self._image)
        self._showing_fit_image = True

    def save_show_fit_all(self, model, args_list):

        plt.close()
        avg_pars = model.get_args()

        if model is not None:
            plot_model = model.copy()
        else:
            plot_model = self._current_model

        num_sets = len(self._data_handlers)
        abs_minX, abs_minY = 1e5, 1e5
        abs_maxX, abs_maxY = -1e5, -1e5

        sum_len = 0

        fig = plt.figure()
        axes : plt.axes = plt.gca()

        print(self._data_handlers, args_list)

        for idx, (handler, args) in enumerate(zip(self._data_handlers, args_list)) :

            x_points = handler.unlogged_x_data
            y_points = handler.unlogged_y_data
            sigma_x_points = handler.unlogged_sigmax_data
            sigma_y_points = handler.unlogged_sigmay_data

            sum_len += len(x_points)
            smooth_x_for_fit = np.linspace( x_points[0], x_points[-1], 4*len(x_points))
            plot_model.set_args(*args)
            if handler.logx_flag and handler.logy_flag :
                fit_vals = [ plot_model.eval_at(xi, X0 = handler.X0, Y0 = handler.Y0)
                             for xi in smooth_x_for_fit ]
            elif handler.logx_flag :
                fit_vals = [ plot_model.eval_at(xi, X0 = handler.X0) for xi in smooth_x_for_fit ]
            elif handler.logy_flag :
                fit_vals = [ plot_model.eval_at(xi, Y0 = handler.Y0) for xi in smooth_x_for_fit ]
            else:
                fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]

            # col = 255 ** (idx/num_sets) / 255
            # col = math.sqrt(idx / num_sets)
            col = idx / num_sets
            print(f"{col=}")
            set_color = (col,col,col)
            axes.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color=set_color)
            plt.plot(smooth_x_for_fit, fit_vals, '-', color=set_color)

            min_X, max_X = min(x_points), max(x_points)
            min_Y, max_Y = min(y_points), max(y_points)

            if min_X < abs_minX :
                abs_minX = min_X
            if min_Y < abs_minY :
                abs_minY = min_Y
            if max_X > abs_maxX :
                abs_maxX = max_X
            if max_Y > abs_maxY :
                abs_maxY = max_Y

            plt.draw()

        # also add average fit
        plot_model.set_args(*avg_pars)
        smooth_x_for_fit = np.linspace(abs_minX, abs_maxX, sum_len)
        if self.data_handler.logx_flag and self.data_handler.logy_flag:
            fit_vals = [plot_model.eval_at(xi, X0=self.data_handler.X0, Y0=self.data_handler.Y0)
                        for xi in smooth_x_for_fit]
        elif self.data_handler.logx_flag:
            fit_vals = [plot_model.eval_at(xi, X0=self.data_handler.X0) for xi in smooth_x_for_fit]
        elif self.data_handler.logy_flag:
            fit_vals = [plot_model.eval_at(xi, Y0=self.data_handler.Y0) for xi in smooth_x_for_fit]
        else:
            fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]
        plt.plot(smooth_x_for_fit, fit_vals, '-', color='r')

        fig.patch.set_facecolor((112 / 255, 146 / 255, 190 / 255))

        plt.xlabel(self.data_handler.x_label)
        plt.ylabel(self.data_handler.y_label)

        if axes.get_xlim()[0] > 0:
            axes.set_xlim([0, axes.get_xlim()[1]])
        elif axes.get_xlim()[1] < 0:
            axes.set_xlim([axes.get_xlim()[0], 0])
        if axes.get_ylim()[0] > 0:
            axes.set_ylim([0, axes.get_ylim()[1]])
        elif axes.get_ylim()[1] < 0:
            axes.set_ylim([axes.get_ylim()[0], 0])

        if self.data_handler.logx_flag:
            log_min, log_max = math.log(abs_minX), math.log(abs_maxX)
            axes.set_xlim([math.exp(log_min - (log_max - log_min) / 10), math.exp(log_max + (log_max - log_min)/10)])
            axes.set(xscale="log")
            axes.spines['right'].set_visible(False)
        else:
            axes.set(xscale="linear")
            axes.spines['left'].set_position(  ('data', 0.) )
            axes.spines['right'].set_position( ('data', 0.) )
            axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
        if self.data_handler.logy_flag:
            axes.set(yscale="log")
            log_min, log_max = math.log(abs_minY), math.log(abs_maxY)
            axes.set_ylim([math.exp(log_min - (log_max-log_min)/10), math.exp(log_max + (log_max-log_min)/10)])
            axes.spines['top'].set_visible(False)
        else:
            axes.set(yscale="linear")
            axes.spines['top'].set_position(    ('data', 0.) )
            axes.spines['bottom'].set_position( ('data', 0.) )
            axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
        axes.set_facecolor((112 / 255, 146 / 255, 190 / 255))


        #  tx is the proportion between xmin and xmax where the zero lies
        # x(tx) = xmin + (xmax - xmin)*tx with 0<tx<1 so
        tx = max(0,-abs_minX / (abs_maxX - abs_minX))
        ty = max(0,-abs_minY / (abs_maxY - abs_minY))
        offset_X, offset_Y = -0.1, 0.0  # how much of the screen is taken by the x and y spines

        axes.xaxis.set_label_coords( 1.050,offset_Y+ty)
        axes.yaxis.set_label_coords( offset_X+tx,+0.750)

        plt.tight_layout()
        plt.savefig(self._image_path)

        # change the view to show the fit as well
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure(image=self._image)
        self._showing_fit_image = True
        self._showing_fit_all_image = True


    def which5_dropdown_trace(self,*args):
        # TODO : should print out the model tree when you change to a different top5 model
        which5_choice = self._which5_name_tkvar.get()
        print(f"Changed top5_dropdown to {which5_choice}")
        # show the fit of the selected model
        rx_sqr, model_name = regex.split(f" ", which5_choice)
        try :
            selected_model_idx = self._optimizer.top5_names.index(model_name)
        except ValueError :
            print(f"{model_name=} is not in {self._optimizer.top5_names}")
            selected_model_idx = 0
        self._current_model = self._optimizer.top5_models[selected_model_idx]
        self._current_args = self._optimizer.top5_args[selected_model_idx]
        self._current_uncs = self._optimizer.top5_uncertainties[selected_model_idx]
        self.save_show_fit_image()
        self.print_results_to_console()

    def create_default_checkboxes(self):

        # still to add:
        # add-custom-primitive textbox
        # fit data / vs / search models on Procedural
        # sliders for initial parameter guesses
        # own model input as textbox

        for idx, name in enumerate(self._checkbox_names_list) :

            checkbox = tk.Checkbutton(
                master=self._gui.children['!frame2'].children['!frame3'],
                text=name,
                variable=self._use_func_dict_name_tkVar[name],
                onvalue=True,
                offvalue=False,
                command=self.checkbox_on_off_command
            )
            checkbox.grid(row=idx+1, column=0, sticky='w')

    def hide_default_checkboxes(self):
        if self._new_user_stage % 31 != 0 :
            # the checkboxes haven't been created yet, no need to hide them
            return
        checkbox0 = self._gui.children['!frame2'].children['!frame3'].children['!checkbutton']
        checkbox0.grid_forget()
        for idx in range(len(self._use_func_dict_name_tkVar)-1):
            checkbox_n = self._gui.children['!frame2'].children['!frame3'].children[f"!checkbutton{idx+2}"]
            checkbox_n.grid_forget()

    def show_default_checkboxes(self):
        checkbox0 = self._gui.children['!frame2'].children['!frame3'].children['!checkbutton']
        checkbox0.grid(row=1, column=0, sticky='w')
        for idx in range(len(self._use_func_dict_name_tkVar)-1):
            checkbox_n = self._gui.children['!frame2'].children['!frame3'].children[f"!checkbutton{idx+2}"]
            checkbox_n.grid(row=idx+2, column=0, sticky='w')

    def checkbox_on_off_command(self):
        pass

    def create_depth_up_down_buttons(self):

        depth_text = tk.Label(
            master = self._gui.children['!frame2'].children['!frame5'],
            text = f"Depth: {self._max_functions_tkInt.get()}"
        )
        down_button = tk.Button( self._gui.children['!frame2'].children['!frame5'],
                                 text = "\U0001F847",
                                 command = self.depth_down_command
                               )
        up_button = tk.Button( self._gui.children['!frame2'].children['!frame5'],
                                  text = "\U0001F845",
                                  command = self.depth_up_command
                                )

        depth_text.grid(row=0, column=0, sticky='w')
        down_button.grid(row=0, column=1, padx=(5,0), pady=5, sticky='w')
        up_button.grid(row=0, column=2, sticky='w')

    # hides their frame, but same idea
    def hide_depth_buttons(self):
        if self._new_user_stage % 31 != 0 :
            # the depth buttons haven't been created yet, no need to hide them
            return
        self._gui.children['!frame2'].children['!frame5'].grid_forget()
    def show_depth_buttons(self):
        self._gui.children['!frame2'].children['!frame5'].grid(row = 4, column=0, sticky = 'w')

    def depth_down_command(self) :
        if self._max_functions_tkInt.get() > 1 :
            self._max_functions_tkInt.set( self._max_functions_tkInt.get() - 1 )
        else :
            self.add_message( f"> Must have a depth of at least 1\n" )
        depth_label = self._gui.children['!frame2'].children['!frame5'].children['!label']
        depth_label.configure(text=f"Depth: {self._max_functions_tkInt.get()}")

    def depth_up_command(self):
        if self._max_functions_tkInt.get() < 7 :
            self._max_functions_tkInt.set( self._max_functions_tkInt.get() + 1 )
        else :
            self.add_message( f"> Cannot exceed a depth of 7\n" )
        depth_label : tk.Label = self._gui.children['!frame2'].children['!frame5'].children['!label']
        depth_label.configure(text=f"Depth: {self._max_functions_tkInt.get()}")

    def create_pause_button(self):
        pause_button = tk.Button( self._gui.children['!frame2'].children['!frame3'],
                                  text = "GO!",
                                  command = self.pause_command
                               )
        pause_button.grid(row=0, column=2, padx=(5,0), sticky='w')
    def hide_pause_button(self):
        if self._new_user_stage % 37 != 0 :
            # the pause button hasn't been created yet, no need to hide it
            return
        print(f"Hide pause button: {self._gui.children['!frame2'].children['!frame3'].children}")
        pause_button = self._gui.children['!frame2'].children['!frame3'].children['!button']
        pause_button.grid_forget()
    def show_pause_button(self):
        pause_button = self._gui.children['!frame2'].children['!frame3'].children['!button']
        pause_button.grid(row=0, column=2, padx=(5,0), pady=5, sticky='w')
    def update_pause_button(self):
        pause_button : tk.Button = self._gui.children['!frame2'].children['!frame3'].children['!button']
        if self.brute_forcing :
            pause_button.configure(text="Pause")
        else :
            pause_button.configure(text="Go")
    def pause_command(self):
        if self.brute_forcing :
            self.brute_forcing = False
            shortpath = regex.split("/", self._filepaths[self._curr_image_num])[-1]
            self.add_message(f"\n \n> For {shortpath} \n")
            self.print_results_to_console()
        else:
            self.brute_forcing = True
            # it's actually a go button
            self.begin_brute_loop()




    """

    Right Panel

    """

    def create_right_panel(self):
        self._gui.columnconfigure(2, minsize=700, weight=1)  # image panel
        column3_frame = tk.Frame(master=self._gui, bg='black')
        column3_frame.grid(row=0, column=2, sticky='news')
        self.add_message("> Welcome to MIW's AutoFit!")

    ##
    #
    # Frames
    #
    ##

    ##
    #
    # Buttons
    #
    ##

    ##
    #
    # Button Commands
    #
    ##

    ##
    #
    # Helper Functions
    #
    ##

    def add_message(self, message_string):

        # TODO: consider also printing to a log file

        text_frame = self._gui.children['!frame3']
        text_frame.update()

        MAX_MESSAGE_LENGTH = 20
        for line in regex.split( f"\n", message_string) :
            if line == "" :
                continue
            new_message_label = tk.Label(master=text_frame, text=line, bg="black", fg="green", font=("consolas",12))
            new_message_label.grid(row=self._num_messages_ever, column=0, sticky=tk.W)
            self._num_messages += 1
            self._num_messages_ever += 1

            new_message_label.update()
            MAX_MESSAGE_LENGTH = floor(text_frame.winfo_height() / new_message_label.winfo_height())


        if self._num_messages > MAX_MESSAGE_LENGTH :
            self.remove_n_messages( self._num_messages - MAX_MESSAGE_LENGTH )

    def remove_n_messages(self, n):

        text_frame = self._gui.children['!frame3']

        key_removal_list = []
        for key in text_frame.children.keys() :
            key_removal_list.append( key )
            if len(key_removal_list) == n :
                break

        for key in key_removal_list:
            text_frame.children[key].destroy()

        self._num_messages = len(text_frame.children)


    """
    
    Frontend functions
    
    """

    @staticmethod
    def get_package_path():

        # keep stepping back from the current directory until we are in the directory /autofit
        loc = os.path.dirname(os.path.abspath(__file__))

        while loc[-7:] != "autofit":
            loc = os.path.dirname(loc)
            if loc == os.path.dirname(loc):
                print(f"""Frontend init: python script {__file__} is not in the AutoFit package's directory.""")

        return loc

    @property
    def normalized_histogram_flag(self):
        return self._normalized_histogram_flags[self._curr_image_num]
    @normalized_histogram_flag.setter
    def normalized_histogram_flag(self, val):
        self._normalized_histogram_flags[self._curr_image_num] = val
    @property
    def data_handler(self):
        return self._data_handlers[self._curr_image_num]
    def show_current_data(self):
        self.show_data(self._curr_image_num)
        self._showing_fit_image = False
        self._showing_fit_all_image = False
    @property
    def brute_forcing(self):
        return self._brute_forcing.get()
    @brute_forcing.setter
    def brute_forcing(self, val):
        self._brute_forcing.set(val)

    def use_functions_dict(self):
        return { key : tkBoolVar.get() for key, tkBoolVar in self._use_func_dict_name_tkVar.items() }
    def max_functions(self):
        return self._max_functions_tkInt.get()

    def exist(self):
        self._gui.mainloop()

