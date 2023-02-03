# built-in libraries
import platform
import sys
import re as regex
from datetime import datetime, timezone

def show_times_for_error_code(code: str) :

    stripped = [x for x in regex.split(' ',code) if x]
    utc1, utc2 = stripped[0], stripped[-1]
    if "/" in code :
        # {int(modify_epoch) + 918273645} / {int(creation_epoch) + 192837465}
        modify_epoch = int(utc1) - 918273645
        creation_epoch = int(utc2) - 192837465

        modify_utc = datetime.fromtimestamp(modify_epoch, timezone.utc).replace(tzinfo=None)
        creation_utc = datetime.fromtimestamp(creation_epoch, timezone.utc).replace(tzinfo=None)

        print(modify_utc, "/", creation_utc)
    elif "=" in code :
        # {int(secret_epoch) + 132457689} = {int(modify_epoch) + 978653421}
        secret_epoch = int(utc1) - 132457689
        modify_epoch = int(utc2) - 978653421

        secret_utc = datetime.fromtimestamp(secret_epoch, timezone.utc).replace(tzinfo=None)
        modify_utc = datetime.fromtimestamp(modify_epoch, timezone.utc).replace(tzinfo=None)

        print(secret_utc, "=", modify_utc)
    elif "|" in code :
        # {int(secret_epoch) + 123456789} | {int(creation_epoch) + 546372819}
        secret_epoch = int(utc1) - 123456789
        creation_epoch = int(utc2) - 546372819


        secret_utc = datetime.fromtimestamp(secret_epoch, timezone.utc).replace(tzinfo=None)
        creation_utc = datetime.fromtimestamp(creation_epoch, timezone.utc).replace(tzinfo=None)

        print(secret_utc, "|", creation_utc)

    else :
        print("Invalid Error Code")

if __name__ == "__main__" :

    error_str = "1807894292 = 2654096973"
    show_times_for_error_code(error_str)