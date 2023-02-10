
# built-in libraries
import _tkinter
import platform
import sys
import os
import re as regex

# external libraries
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk

# internal classes
from autofit.src.package import pkg_path, logger

class Validator:

    def __init__(self):
        self._filepath = pkg_path()
        if sys.platform == "darwin" :
            self._filepath += "/libdscheme.5.dylib"
        else :
            self._filepath += "/libdscheme.H3UN78J69H7J8K9JAS76KP8KLFSAHT.gfortran-win_amd64.dll"

    @staticmethod
    def extract_epoch_from_file(filepath) -> tuple[int, str]:
        if not os.path.exists(filepath) :
            logger("1> No file detected, exiting...")
            return 0, f"Error code 1, exiting..."
        try:
            with open(filepath) as file :
                cipher = ""
                for line in file:
                    if len(line) < 5 :
                        logger("2> No content, exiting...")
                        return 0, f"Error code 2, exiting..."
                    cipher = line
                    break
                if cipher == "" :
                    logger("3> No content, exiting...")
                    return 0, f"Error code 3, exiting..."
                message : str = Validator.de_crypt(cipher)
        except FileNotFoundError :
            logger("4> No file detected, exiting...")
            return 0, f"Error code 4, exiting..."

        part : list[str] = regex.split(f"<<<",message)
        try :
            _ , str_epoch = regex.split(f">>>",part[0])
            _, str_transaction_ID = regex.split(f">>>", part[1])
        except ValueError:
            logger("5> Invalid hash, exiting...")
            return 0, f"Error code 5, exiting..."

        # if the file doesn't decrypt, it's been modified
        try :
            secret_epoch = int(str_epoch)
        except ValueError:
            logger("6> Invalid int, exiting...")
            return 0, f"Error code 6, exiting..."

        return secret_epoch, ""

    def invalid_config(self) -> str:

        from datetime import datetime, timezone

        # when the secret was made in UTC (not signed)
        secret_epoch, err_str = Validator.extract_epoch_from_file(self._filepath)
        if secret_epoch < 1 or err_str != "" :
            return err_str

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
                logger("11> Linux isn't supported.")
                return f"Error code 11, exiting..."

        zero_utc = datetime.fromtimestamp( 0, timezone.utc ).replace(tzinfo=None)
        creation_utc = datetime.fromtimestamp( creation_epoch, timezone.utc ).replace(tzinfo=None)
        modify_utc = datetime.fromtimestamp( modify_epoch )
        secret_utc = datetime.fromtimestamp( secret_epoch, timezone.utc ).replace(tzinfo=None)

        # logger("Validator: create -- ",creation_utc)
        # logger("Validator: modify -- ",modify_utc)
        # logger("Validator: secret -- ",secret_utc)

        # logger(creation_epoch, (creation_utc-zero_utc).total_seconds() )
        # logger(modify_epoch, (modify_utc-zero_utc).total_seconds() )
        # logger(secret_epoch, (secret_utc-zero_utc).total_seconds() )

        creation_epoch =  (creation_utc-zero_utc).total_seconds()
        modify_epoch = (modify_utc-zero_utc).total_seconds()
        secret_epoch = (secret_utc-zero_utc).total_seconds()

        # logger(creation_utc,modify_utc,secret_utc)

        seconds_cm = (creation_utc-modify_utc).total_seconds()
        seconds_cs = (creation_utc-secret_utc).total_seconds()
        seconds_ms = (modify_utc-secret_utc).total_seconds()

        # logger(seconds_cm,seconds_cs,seconds_ms,seconds_mc)

        # assume that the download (modify_epoch) to install (unzipping, creation_time) will take less than an hour
        if abs(seconds_cm) > 60*60 + 1 :
            # logger(modify_epoch, creation_epoch)
            logger("7> ")  # You need to have less time between downloading and unzipping the file.
            # return f"Error code {modify_time} / {creation_time}, exiting..."
            return f"Error code {int(modify_epoch) + 918273645} / {int(creation_epoch) + 192837465}, exiting..."
        # assume that the hidden secret_epoch and the modify_epoch (download) are aligned
        if abs(seconds_ms) > 5 :
            # logger(secret_epoch, modify_epoch)
            logger("8>")  # The secret file has been modified.
            # return f"Error code {secret_time} = {modify_time}, exiting..."
            return f"Error code {int(secret_epoch) + 132457689} = {int(modify_epoch) + 978653421}, exiting..."
        # assume that the hidden secret_epoch and the creation_time (unzipping) are less than an hour apart
        if abs(seconds_cs) > 60*60 + 1 :
            # logger(secret_epoch, creation_epoch)
            logger("9>")  # You need to have less time between downloading and unzipping the file.
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
        try :
            gui.iconbitmap(f"{pkg_path()}/icon.ico")
        except _tkinter.TclError :
            messagebox.showerror(f"Packaging error: no icon.ico located in {pkg_path()}")
        if sys.platform == "darwin" :
            try :
                iconify = Image.open(f"{pkg_path()}/splash.png")
                photo = ImageTk.PhotoImage(iconify)
                gui.iconphoto(False,photo)
            except _tkinter.TclError:
                messagebox.showerror(f"Packaging error: no splash.png located in {pkg_path()}")
        gui.title("MIW's AutoFit")

        # logger(pkg_path())
        messagebox.showerror(f"Configuration Error in {pkg_path()}\n\n",
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

        # logger("".join(copy), cipher)
        return "".join(copy)


