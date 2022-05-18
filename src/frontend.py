# default libraries
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
from autofit.src.file_handler import FileHandler
from autofit.src.optimizer import Optimizer

class Frontend:

    def __init__(self):

        # UX
        self._new_user_stage = 1

        # UI
        self._gui = tk.Tk()
        self._os_width = self._gui.winfo_screenwidth()
        self._os_height = self._gui.winfo_screenheight()

        # rendering
        self._curr_image_num = -1
        self._image_path = None
        self._image = None
        self._image_frame = None

        # file handling
        self._filepaths = []
        self._file_handlers = []

        # messaging
        self._num_messages_ever = 0
        self._num_messages = 0
        self._full_flag = 0

        # backend connections
        self._optimizer = None

        # load in
        self.load_splash_screen()

        # exist
        self._gui.mainloop()

    def load_splash_screen(self):

        gui = self._gui

        # window size and title
        gui.geometry(f"{round(self._os_width*3/4)}x{round(self._os_height*3/4)}")
        gui.title("AutoFit")

        # icon and splash image
        loc = Frontend.get_package_path()
        gui.iconbitmap(f"{loc}/icon.ico")

        gui.rowconfigure(0, minsize=800, weight=1)

        # column 1 -- buttons

        column1_frame = tk.Frame(master=gui, relief=tk.RAISED, bg='white')
        column1_frame.grid(row=0, column=0, sticky='ns')

        load_data_button = tk.Button(
            master = column1_frame,
            text = "Load Data",
            command = self.open_file_command
        )
        load_data_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # column 2 -- show data
        gui.columnconfigure(1, minsize=400, weight=1)       # image panel

        column2_frame = tk.Frame(master=gui, relief=tk.RIDGE)
        column2_frame.grid(row=0, column=1, sticky='news')

        self._image_path = f"{loc}/splash.png"
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame = tk.Label( master=column2_frame,
                                image=self._image ,
                                relief=tk.SUNKEN  )
        self._image_frame.grid(row=0, column=0)


        # column 3 -- text and options
        gui.columnconfigure(2, minsize=400, weight=1)       # image panel

        column3_frame = tk.Frame(master=gui, bg='black')
        column3_frame.grid(row=0, column=2, sticky='news')

        self.add_message("> Welcome to MIW's AutoFit!")

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

        print(f"Removing{key_removal_list}")

        print(f"First {len(text_frame.children)}")
        for key in key_removal_list:
            text_frame.children[key].destroy()
        print(f"Then {len(text_frame.children)}")

        self._num_messages = len(text_frame.children)

        print(text_frame.children)



    def open_file_command(self):
        loc = Frontend.get_package_path()

        new_filepaths = fd.askopenfilenames(initialdir=f"{loc}/data", title="Select a file to fit",
                                              filetypes=(("comma-separated files", "*.csv"),
                                                         ("text files", "*.txt"),
                                                         ("Excel files", "*.xls*"))
                                              )
        self._filepaths.extend(new_filepaths)
        if self._new_user_stage % 2 != 0 :
            self.add_fit_button()
            self._new_user_stage *= 2

        if len(new_filepaths) > 0 :
            self._curr_image_num = len(self._file_handlers)
            self.load_new_data(new_filepaths)
            self.show_data(self._curr_image_num)
            if self._new_user_stage % 3 != 0 :
                self.add_data_perusal()
                self._new_user_stage *= 3
            print(f"Loaded {len(new_filepaths)} files.")

        if len(self._filepaths) > 1 and self._new_user_stage % 5 != 0:
            self.add_data_select_left_right()
            self._new_user_stage += 1


    def load_new_data(self, new_filepaths_lists):
        for path in new_filepaths_lists :
            self._file_handlers.append(FileHandler(filepath=path))
    def reload_all_data(self):
        self._file_handlers = []
        for path in self._filepaths :
            self._file_handlers.append(FileHandler(filepath=path))

    def show_data(self, file_num=0):

        mod_file_num = file_num % len(self._file_handlers)

        new_image_path = f"{Frontend.get_package_path()}/plots/front_end_current_plot.png"
        # create a scatter plot of the first file

        x_points = []
        y_points = []
        sigma_x_points = []
        sigma_y_points = []

        for datum in self._file_handlers[mod_file_num ].data:
            x_points.append(datum.pos)
            y_points.append(datum.val)
            sigma_x_points.append(datum.sigma_pos)
            sigma_y_points.append(datum.sigma_val)

        plt.close()
        fig = plt.figure()
        fig.patch.set_facecolor( (112/255, 146/255, 190/255) )
        plt.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color='k')
        plt.xlabel(self._file_handlers[mod_file_num].x_label)
        plt.ylabel(self._file_handlers[mod_file_num].y_label)
        axes = plt.gca()
        axes.set_facecolor( (112/255, 146/255, 190/255) )
        plt.savefig(new_image_path)

        # replace the splash graphic with the plot
        self._image_path = new_image_path
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure( image = self._image )

    def add_data_perusal(self):

        inspection_frame = tk.Frame(
            master=self._gui.children['!frame2'],
        )
        inspection_frame.grid(row=1, column=0, sticky='w')

        data_perusal_button = tk.Button(
            master = inspection_frame,
            text = "Inspect",
            command = plt.show
        )
        data_perusal_button.grid(row=0, column=0, padx=5, pady=5)

    def show_left(self):
        self._curr_image_num -= 1
        self.show_data(self._curr_image_num)
        self.update_data_select()

    def show_right(self):
        self._curr_image_num += 1
        self.show_data(self._curr_image_num)
        self.update_data_select()

    def update_data_select(self):
        inspection_frame = self._gui.children['!frame2'].children['!frame']
        text_label = inspection_frame.children['!label']

        text_label.configure(text=f"{(self._curr_image_num % len(self._file_handlers)) + 1 }/{len(self._file_handlers)}")

    def add_data_select_left_right(self):

        inspection_frame = self._gui.children['!frame2'].children['!frame']


        left_button = tk.Button( master = inspection_frame,
                                 text = "\U0001F844",
                                 command = self.show_left
                               )
        count_text = tk.Label(
            master=inspection_frame,
            text = f"{self._curr_image_num % len(self._file_handlers) + 1}/{len(self._file_handlers)}"
        )
        right_button = tk.Button( master = inspection_frame,
                                  text = "\U0001F846",
                                  command = self.show_right
                                )
        left_button.grid(row=0, column=1, padx=5, pady=5)
        count_text.grid(row=0, column=2)
        right_button.grid(row=0, column=3, padx=5, pady=5)

    def fit_single_data_command(self):
        print("Fitting data")

        # FInd the fit for the currently displayed data
        self._optimizer = Optimizer(data=self._file_handlers[self._curr_image_num].data)
        self._optimizer.fit_single_data_set()
        self._optimizer.save_fit(self._image_path)

        # change the view to show the fit as well
        self._image = ImageTk.PhotoImage(Image.open(self._image_path))
        self._image_frame.configure(image=self._image)

        # add fit all button if there's more than one file
        if self._new_user_stage % 11 != 0 and len(self._file_handlers) > 1 :
            self.add_fit_all_button()
            self._new_user_stage *= 11

        # print out the parameters on the right
        print_string = ""
        print_string += f" \n> Optimal model is {self._optimizer.best_model} with\n"
        for idx, (par, unc) in enumerate( zip(self._optimizer.parameters, self._optimizer.uncertainties) ):
            print_string += f"  c{idx} = {par:.2E}  +-  {unc:.2E}\n"
        print_string += "\n>  As a tree, this is \n"
        print_string += self._optimizer.best_model.tree_as_string_with_args()
        self.add_message(print_string)

        # add buttons to adjust fit options
        if self._new_user_stage % 7 != 0 :
            self._new_user_stage *= 7
            self.add_fit_options_frame()


    def add_fit_button(self):

        # add to column 1 -- buttons
        fit_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Fit Data",
            command = self.fit_single_data_command
        )
        fit_data_button.grid(row=1, column=0, sticky="ew", padx=5)

    def add_fit_all_button(self):

        # add to column 1 -- buttons
        load_data_button = tk.Button(
            master = self._gui.children['!frame'],
            text = "Fit All",
            command = self.fit_every_file_command
        )
        load_data_button.grid(row=2, column=0, sticky="ew", padx=5)


    def fit_every_file_command(self):
        pass

    def add_fit_options_frame(self):
        middle_panel = self._gui.children['!frame2']

        fit_options_frame = tk.Frame(master=middle_panel)
        fit_options_frame.grid(row = 2, column=0, sticky='w')

        # use trig/exp/powers checkmarks
        # add-custom-primitive textbox
        # search depth

        # top 5 fits quick list
        # sliders for initial parameter guesses
        # own model input as textbox



    @staticmethod
    def get_package_path():

        # keep stepping back from the current directory until we are in the directory /autofit
        loc = os.path.dirname(os.path.abspath(__file__))

        while loc[-7:] != "autofit":
            loc = os.path.dirname(loc)
            if loc == os.path.dirname(loc):
                print(f"""Frontend init: python script {__file__} is not in the AutoFit package's directory.""")

        return loc

    def exist(self):
        self._gui.mainloop()

