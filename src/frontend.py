# default libraries
import math
from dataclasses import field
from math import floor

# external libraries
import tkinter as tk
import tkinter.filedialog as fd

import matplotlib.pyplot as plt
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

        # file handling
        self._filepaths = []
        self._data_handlers = []

        # messaging
        self._num_messages_ever = 0
        self._num_messages = 0
        self._full_flag = 0

        # backend connections
        self._optimizer = None   # Optimizer
        self._model_name = None  # tk.StringVar

        # load in
        self.load_splash_screen()

        # exist
        self._gui.mainloop()

    # create left, right, and middle panels
    def load_splash_screen(self):

        gui = self._gui

        # window size and title
        gui.geometry(f"{round(self._os_width*3/4)}x{round(self._os_height*3/4)}")
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

        # add to column 1 -- buttons
        fit_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Fit Data",
            command = self.fit_data_command
        )
        fit_data_button.grid(row=1, column=0, sticky="ew", padx=5)

    def create_fit_all_button(self):

        # add to column 1 -- buttons
        load_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Fit All",
            command = self.fit_every_file_command
        )
        load_data_button.grid(row=2, column=0, sticky="ew", padx=5)

    ##
    #
    # Button commands
    #
    ##

    def load_data_command(self):
        loc = Frontend.get_package_path()

        new_filepaths = list( fd.askopenfilenames(initialdir=f"{loc}/data", title="Select a file to fit",
                                                  filetypes=(("comma-separated files", "*.csv"),
                                                            ("text files", "*.txt"),
                                                            ("Excel files", "*.xls*"))
                                                 )
                            )
        # trim duplicates
        for path in new_filepaths[:] :
            if path in self._filepaths :
                shortpath = regex.split( f"/", path )[-1]
                print(f"{shortpath} already loaded")
                new_filepaths.remove(path)
        for path in new_filepaths :
            self._filepaths.append(path)
            self._normalized_histogram_flags.append(False)

        if self._new_user_stage % 2 != 0 :
            self.create_fit_button()
            self._new_user_stage *= 2

        if len(new_filepaths) > 0 :
            self._curr_image_num = len(self._data_handlers)
            self.load_new_data(new_filepaths)
            self.show_current_data()
            if self._new_user_stage % 3 != 0 :
                self.create_inspect_button()
                self._new_user_stage *= 3
            print(f"Loaded {len(new_filepaths)} files.")

        if len(self._filepaths) > 1 and self._new_user_stage % 5 != 0:
            self.create_left_right_buttons()
            self._new_user_stage *= 5

    ##
    #
    # Helper functions
    #
    ##

    def load_new_data(self, new_filepaths_lists):
        for path in new_filepaths_lists :
            self._data_handlers.append(DataHandler(filepath=path))

    def reload_all_data(self):
        self._data_handlers = []
        for path in self._filepaths :
            self._data_handlers.append(DataHandler(filepath=path))

    def show_data(self, file_num=0):

        mod_file_num = file_num % len(self._data_handlers)

        new_image_path = f"{Frontend.get_package_path()}/plots/front_end_current_plot.png"
        # create a scatter plot of the first file

        x_points = []
        y_points = []
        sigma_x_points = []
        sigma_y_points = []

        for datum in self.data_handler.data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            sigma_x_points.append(datum.sigma_pos)
            sigma_y_points.append(datum.sigma_val)

        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
        plt.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        plt.xlabel(self.data_handler.x_label)
        plt.ylabel(self.data_handler.y_label)
        axes = plt.gca()
        if axes.get_xlim()[0] > 0 :
            axes.set_xlim( [0, axes.get_xlim()[1]] )
        if axes.get_ylim()[0] > 0 :
            axes.set_ylim( [0, axes.get_ylim()[1]] )
        axes.set_facecolor( (112/255, 146/255, 190/255) )

        plt.savefig(new_image_path)

        # replace the splash graphic with the plot
        self._image_path = new_image_path
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure( image = self._image )

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
                self.show_normalize_button()
        else:
            self.hide_normalize_button()



    def fit_data_command(self):

        # add buttons to adjust fit options
        if self._new_user_stage % 7 != 0 :
            print(self._new_user_stage)
            self._new_user_stage *= 7
            self.create_function_dropdown()

        # Find the fit for the currently displayed data
        data = self.data_handler.data
        self._optimizer = Optimizer(data=data)
        plot_model = None
        if self._model_name.get() == "Linear" :
            print("Fitting to linear model")
            plot_model = CompositeFunction.built_in("Linear")
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model)
        elif self._model_name.get() == "Gaussian" and self.normalized_histogram_flag:
            print("Fitting to Normal distribution")
            plot_model = CompositeFunction.built_in("Normal")
            init_guess_mean = ( max( [datum.pos for datum in data] ) + min( [datum.pos for datum in data] ) ) / 2
            init_guess_sigma = ( max( [datum.pos for datum in data] ) - min( [datum.pos for datum in data] ) ) / 4
            initial_guess = [ -1/(2*init_guess_sigma**2),-init_guess_mean]
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model,initial_guess=initial_guess)
        elif self._model_name.get() == "Gaussian" :
            print("Fitting to Gaussian model")
            plot_model = CompositeFunction.built_in("Gaussian")
            init_guess_amplitude = max( [datum.val for datum in data] )
            init_guess_mean = ( max( [datum.pos for datum in data] ) + min( [datum.pos for datum in data] ) ) / 2
            init_guess_sigma = ( max( [datum.pos for datum in data] ) - min( [datum.pos for datum in data] ) ) / 4
            initial_guess = [ init_guess_amplitude, -1/(2*init_guess_sigma**2),-init_guess_mean]
            self._optimizer.parameters_and_uncertainties_from_fitting(plot_model,initial_guess=initial_guess)
        elif self._model_name.get() == "Procedural":
            print("Fitting to procedural model")
            # find fit button should now be find model
            self._optimizer.find_best_model_for_dataset()
            plot_model = self._optimizer.best_model
        else:
            pass
        self._optimizer.save_fit_image(self._image_path,
                                       x_label=self.data_handler.x_label,
                                       y_label=self.data_handler.y_label,
                                       model=plot_model)


        # change the view to show the fit as well
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure(image=self._image)

        # add fit all button if there's more than one file
        if self._new_user_stage % 11 != 0 and len(self._data_handlers) > 1 :
            self.create_fit_all_button()
            self._new_user_stage *= 11

        # print out the parameters on the right
        shortpath = regex.split("/", self._filepaths[self._curr_image_num])[-1]
        print_string = f"\n \n> For {shortpath} \n"
        if self._model_name.get() == "Linear" :
            print_string += f"  Linear fit is y = m x + b with\n"
            m, sigmam = self._optimizer.parameters[0], self._optimizer.uncertainties[0]
            b = self._optimizer.parameters[0]*self._optimizer.parameters[1]
            sigmab = math.sqrt( self._optimizer.uncertainties[0]**2 * self._optimizer.parameters[1]**2 +
                                self._optimizer.uncertainties[1]**2 * self._optimizer.parameters[0]**2   )
            print_string += f"   m = {m:.2E}  +-  {sigmam:.2E}\n"
            print_string += f"   b = {b:.2E}  +-  {sigmab:.2E}\n"
        elif self._model_name.get() == "Gaussian" and self.normalized_histogram_flag:
            print_string += f"  Normal fit is y = 1/sqrt(2pi sigma^2) exp( -(x-mu)^2 / 2 sigma^2 ) with\n"
            mu, sigmamu = -self._optimizer.parameters[1], self._optimizer.uncertainties[1]
            sigma = math.sqrt( -1 / (2*self._optimizer.parameters[0]) )
            sigmasigma = math.sqrt( 1/(4*self._optimizer.parameters[0]**2 * sigma) ) * self._optimizer.uncertainties[0]
            print_string += f"   mu    = {mu:.2E}  +-  {sigmamu:.2E}\n"
            print_string += f"   sigma = {sigma:.2E}  +-  {sigmasigma:.2E}\n"
        elif self._model_name.get() == "Gaussian" :
            print_string += f"  Gaussian fit is y = A exp( -(x-mu)^2 / 2 sigma^2 ) with\n"
            A, sigmaA = self._optimizer.parameters[0], self._optimizer.uncertainties[0]
            mu, sigmamu = -self._optimizer.parameters[2], self._optimizer.uncertainties[2]
            sigma = math.sqrt( -1 / (2*self._optimizer.parameters[1]) )
            sigmasigma = math.sqrt( 1/(4*self._optimizer.parameters[1]**2 * sigma) ) * self._optimizer.uncertainties[1]
            print_string += f"   A     = {A:.2E}  +-  {sigmaA:.2E}\n"
            print_string += f"   mu    = {mu:.2E}  +-  {sigmamu:.2E}\n"
            print_string += f"   sigma = {sigma:.2E}  +-  {sigmasigma:.2E}\n"
        elif self._model_name.get() == "Procedural":
            print_string += f"Optimal model is {self._optimizer.best_model} with\n"
            for idx, (par, unc) in enumerate(zip(self._optimizer.parameters, self._optimizer.uncertainties)):
                print_string += f"  c{idx} = {par:.2E}  +-  {unc:.2E}\n"
            print_string += "\n>  As a tree, this is \n"
            print_string += self._optimizer.best_model.tree_as_string_with_args()
        else:
            pass

        self.add_message(print_string)

            # use trig/exp/powers checkmarks
            # add-custom-primitive textbox
            # search depth

            # top 5 fits quick list
            # sliders for initial parameter guesses
            # own model input as textbox

    def fit_every_file_command(self):
        pass


    """

    Middle Panel

    """

    def create_middle_panel(self):
        self._gui.columnconfigure(1, minsize=720, weight=1)  # image panel
        middle_panel_frame = tk.Frame(master=self._gui, relief=tk.RIDGE)
        middle_panel_frame.grid(row=0, column=1, sticky='news')
        self.create_image_frame()
        self.create_data_perusal_frame()
        self.create_fit_options_frame()
        self.create_plot_options_frame()

    ##
    #
    # Frames
    #
    ##

    def create_image_frame(self):
        image_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        image_frame.grid(row=0, column=0, sticky='w')
        self.load_splash_image()

    def create_data_perusal_frame(self):
        data_perusal_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        data_perusal_frame.grid(row=1, column=0, sticky='w')

    def create_fit_options_frame(self):
        fit_options_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        fit_options_frame.grid(row = 3, column=0, sticky='w')

    def create_plot_options_frame(self):
        self._gui.children['!frame2'].columnconfigure(1,minsize=50)
        plot_options_frame = tk.Frame(
            master=self._gui.children['!frame2']
        )
        plot_options_frame.grid(row = 0, column=1, sticky='ns')


    ##
    #
    # Buttons
    #
    ##

    def create_inspect_button(self):
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

        print("Creating function dropdown")
        self._gui.children['!frame2'].rowconfigure(2, minsize=1)
        black_line_as_frame = tk.Frame(
            master=self._gui.children['!frame2'],
            bg = 'black'
        )
        black_line_as_frame.grid(row = 2, column=0, sticky='ew')

        func_list = ["Linear", "Gaussian", "Procedural"]

        self._model_name = tk.StringVar(self._gui.children['!frame2'].children['!frame2'])
        self._model_name.set( func_list[0] )

        function_dropdown = tk.OptionMenu(
            self._gui.children['!frame2'].children['!frame3'],
            self._model_name,
            *func_list
        )
        function_dropdown.configure(width=9)
        function_dropdown.grid(row=0, column=0)

        self._model_name.trace('w', self.function_dropdown_trace)

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

    def image_left_command(self):
        self._curr_image_num = (self._curr_image_num - 1) % len(self._data_handlers)
        self.show_current_data()
        self.update_data_select()
        if self.data_handler.histogram_flag :
            self.show_normalize_button()
        else :
            self.hide_normalize_button()

    def image_right_command(self):
        self._curr_image_num  = (self._curr_image_num + 1) % len(self._data_handlers)
        self.show_current_data()
        self.update_data_select()
        if self.data_handler.histogram_flag :
            self.show_normalize_button()
        else :
            self.hide_normalize_button()

    def show_residuals_command(self):
        pass

    def normalize_command(self):
        self.data_handler.normalize_histogram_data()
        self.normalized_histogram_flag = True
        self.show_data(self._curr_image_num)

    def logx_command(self):
        pass

    def logy_command(self):
        pass


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
        inspection_frame = self._gui.children['!frame2'].children['!frame2']
        text_label = inspection_frame.children['!label']

        text_label.configure(text=f"{(self._curr_image_num % len(self._data_handlers)) + 1 }/{len(self._data_handlers)}")

    def function_dropdown_trace(self,*args):
        model_choice = self._model_name.get()
        return model_choice

    """

    Right Panel

    """

    def create_right_panel(self):
        self._gui.columnconfigure(2, minsize=700)  # image panel
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

        text_frame = self._gui.children['!frame3']
        text_frame.update()

        MAX_MESSAGE_LENGTH = 20
        for line in regex.split( f"\n", message_string) :
            if line == "" :
                continue
            new_message_label = tk.Label(master=text_frame, text=line, bg="black", fg="green", font="consolas")
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

    def exist(self):
        self._gui.mainloop()

