# external libraries
import tkinter as tk
import tkinter.filedialog as fd

class LeftPanel:

    def __init__(self, master: tk.Tk ):

        # frame for the left panel
        self._frame = tk.Frame(master=master, relief=tk.RAISED, bg='white')
        self._frame.grid(row=0, column=0, sticky='ns')


    """
    
    Button creation
    
    """
    def create_load_data_button(self):
        load_data_button = tk.Button(
            master=self._frame,
            text="Load Data",
            command=self.open_file_command
        )
        load_data_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    """
    
    Button click actions
    
    """