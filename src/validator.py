
# built-in libraries
import platform
import sys
import os as os
import re as regex

# external libraries
from tkinter import messagebox
import tkinter as tk
# user-defined classes


class Validator:

    def __init__(self):
        self._filepath = (Validator.get_package_path()
                          + "/libdscheme.H3UN78J69H7J8K9JAS76KP8KLFSAHT.gfortran-win_amd64.dll")



    def invalid_config(self):

        try:
            with open(self._filepath) as file :
                for line in file:
                    cipher = line
                    break
                message : str = Validator.de_crypt(cipher)
                print(message)
        except FileNotFoundError :
            print("1> No secret file detected, exiting...")
            return 1

        part : list[str] = regex.split(f"<<<",message)
        _ , str_epoch = regex.split(f">>>",part[0])
        _, str_transaction_ID = regex.split(f">>>", part[1])

        # if the file doesn't decrypt, it's been modified
        try :
            epoch = int(str_epoch)
        except :
            print("2> Epoch is not int-like, exiting...")
            return 2

        if platform.system() == "Windows" :
            creation_time = os.path.getctime(self._filepath)  # when the file was unzipped/copied
            modify_time = os.path.getmtime(self._filepath)    # when the file was created on the server / modified by pirate
        else :
            stat = os.stat(self._filepath)
            try :
                modify_time = stat.st_mtime
                creation_time = stat.st_birthtime
            except AttributeError :
                print("11> Linux isn't supported.")
                return 11

        print(creation_time,modify_time,epoch)

        # assume that the download (modify_time) to install (unzipping, creation_time) will take less than an hour
        if abs(modify_time-creation_time) > 60*60 :
            print(modify_time, creation_time)
            print("3> You need to have less time between downloading and unzipping the file.")
            return 3
        # assume that the hidden epoch and the modify_time (download) are aligned
        if abs(epoch-modify_time) > 5 :
            print(epoch, modify_time)
            print("4> The secret file has been modified.")
            return 4
        # assume that the hidden epoch and the creation_time (unzipping) are less than an hour apart
        if abs(epoch-creation_time) > 60*60 :
            print(epoch, creation_time)
            print("5> You need to have less time between downloading and unzipping the file.")
            return 5
        return 0

    @staticmethod
    def invalid_popup():

        gui = tk.Tk()

        # window size and title
        gui.geometry(
            f"{round(gui.winfo_screenwidth() * 5 / 6)}x{round(gui.winfo_screenheight() * 5 / 6)}+5+10")
        gui.rowconfigure(0, minsize=800, weight=1)

        # icon image and window title
        gui.iconbitmap(f"{Validator.get_package_path()}/icon.ico")
        gui.title("AutoFit")

        print(Validator.get_package_path())
        error_box = messagebox.showerror("Configuration Error",
                                         "Please try re-downloading this package from "
                                         "ingliswhalen.com/MIWs-AutoFit/AutoFit-Pro-Downloads")
        raise SystemExit

    @staticmethod
    def de_crypt(cipher: str) -> str:

        copy = [*cipher]
        cipher_len = len(cipher)
        jump = 61

        while cipher_len % jump == 0 :
            jump += 1
        for idx in range(cipher_len) :
            char_scram : int = ( ord(cipher[idx])-3329 ) % 256
            while char_scram < 0 :
                char_scram += 256
            copy[ jump*(idx+1) % cipher_len ] = chr(char_scram)

        print("".join(copy), cipher)
        return "".join(copy)



    @staticmethod
    def get_package_path():

        try:
            # noinspection PyUnresolvedReferences
            return sys._MEIPASS  # for pyinstaller
        except AttributeError:
            pass
            # print("It doesn't know about _MEIPASS")

        # keep stepping back from the current directory until we are in the directory /autofit
        filepath = os.path.abspath(__file__)
        loc = os.path.dirname(filepath)

        while loc[-7:] != "autofit":
            loc = os.path.dirname(loc)
            if loc == os.path.dirname(loc):
                print(f"""Validator init: python script {__file__} is not in the AutoFit package's directory.""")

        return loc