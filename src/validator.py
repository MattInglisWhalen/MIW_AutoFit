
# built-in libraries
import platform
import sys
import os as os
import re as regex

# external libraries
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
# user-defined classes


class Validator:

    def __init__(self):
        self._filepath = (Validator.get_package_path()
                          + "/libdscheme.H3UN78J69H7J8K9JAS76KP8KLFSAHT.gfortran-win_amd64.dll")

    @staticmethod
    def extract_epoch_from_file(filepath):
        try:
            with open(filepath) as file :
                for line in file:
                    cipher = line
                    break
                message : str = Validator.de_crypt(cipher)
                # print(message)
        except FileNotFoundError :
            print("1> No secret file detected, exiting...")
            return f"Error code 1, exiting..."

        part : list[str] = regex.split(f"<<<",message)
        _ , str_epoch = regex.split(f">>>",part[0])
        _, str_transaction_ID = regex.split(f">>>", part[1])

        # if the file doesn't decrypt, it's been modified
        try :
            secret_epoch = int(str_epoch)
        except ValueError:
            print("2> Epoch is not int-like, exiting...")
            return f"Error code 2, exiting..."

        return secret_epoch

    def invalid_config(self) -> str:

        from datetime import datetime, timezone

        secret_epoch = Validator.extract_epoch_from_file(self._filepath)  # when the secret was made in UTC (not signed)

        # this timing is based off the assumption that the ingliswhalen.com server signs the certificate using UTC time
        if platform.system() == "Windows" :
            creation_epoch = os.path.getctime(self._filepath)  # when the file was unzipped/copied (locally signed)
            modify_epoch = os.path.getmtime(self._filepath)    # when the file was created on the server (server signed)
        else :
            stat = os.stat(self._filepath)
            try :
                modify_epoch = stat.st_mtime
                creation_epoch = stat.st_birthtime
            except AttributeError :
                print("11> Linux isn't supported.")
                return f"Error code 11, exiting..."

        zero_utc = datetime.fromtimestamp( 0, timezone.utc ).replace(tzinfo=None)
        creation_utc = datetime.fromtimestamp( creation_epoch, timezone.utc ).replace(tzinfo=None)
        modify_utc = datetime.fromtimestamp( modify_epoch )
        secret_utc = datetime.fromtimestamp( secret_epoch, timezone.utc ).replace(tzinfo=None)

        # print(creation_epoch, (creation_utc-zero_utc).total_seconds() )
        # print(modify_epoch, (modify_utc-zero_utc).total_seconds() )
        # print(secret_epoch, (secret_utc-zero_utc).total_seconds() )

        creation_epoch =  (creation_utc-zero_utc).total_seconds()
        modify_epoch = (modify_utc-zero_utc).total_seconds()
        secret_epoch = (secret_utc-zero_utc).total_seconds()

        # print(creation_utc,modify_utc,secret_utc)

        seconds_cm = (creation_utc-modify_utc).total_seconds()
        seconds_cs = (creation_utc-secret_utc).total_seconds()
        seconds_ms = (modify_utc-secret_utc).total_seconds()

        # print(seconds_cm,seconds_cs,seconds_ms,seconds_mc)

        # assume that the download (modify_epoch) to install (unzipping, creation_time) will take less than an hour
        if abs(seconds_cm) > 60*60 :
            # print(modify_epoch, creation_epoch)
            print("3> You need to have less time between downloading and unzipping the file.")
            # return f"Error code {modify_time} / {creation_time}, exiting..."
            return f"Error code {int(modify_epoch) + 918273645} / {int(creation_epoch) + 192837465}, exiting..."
        # assume that the hidden secret_epoch and the modify_epoch (download) are aligned
        if abs(seconds_ms) > 5 :
            # print(secret_epoch, modify_epoch)
            print("4> The secret file has been modified.")
            # return f"Error code {secret_time} = {modify_time}, exiting..."
            return f"Error code {int(secret_epoch) + 132457689} = {int(modify_epoch) + 978653421}, exiting..."
        # assume that the hidden secret_epoch and the creation_time (unzipping) are less than an hour apart
        if abs(seconds_cs) > 60*60 :
            # print(secret_epoch, creation_epoch)
            print("5> You need to have less time between downloading and unzipping the file.")
            # return f"Error code {secret_time} | {creation_time}, exiting..."
            return f"Error code {int(secret_epoch) + 123456789} | {int(creation_epoch) + 546372819}, exiting..."
        return ""

    @staticmethod
    def invalid_popup(error_msg : str):

        gui = tk.Tk()

        # window size and title
        gui.geometry(
            f"{round(gui.winfo_screenwidth() * 5 / 6)}x{round(gui.winfo_screenheight() * 5 / 6)}+5+10")
        gui.rowconfigure(0, minsize=800, weight=1)

        # icon image and window title
        gui.iconbitmap(f"{Validator.get_package_path()}/icon.ico")
        if sys.platform == "darwin" :
            iconify = Image.open(f"{Validator.get_package_path()}/splash.png")
            photo = ImageTk.PhotoImage(iconify)
            gui.iconphoto(False,photo)
        gui.title("MIW's AutoFit")

        # print(Validator.get_package_path())
        messagebox.showerror("Configuration Error",
                                         f"{error_msg}\n\nPlease try re-downloading this package from "
                                         f"ingliswhalen.com/MIWs-AutoFit/AutoFit-Pro-Downloads")
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

        # print("".join(copy), cipher)
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

