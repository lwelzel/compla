import re
import os
import sys
import platform
from pathlib import Path




def get_wdir_ddir():
    os_name = platform.system()

    WDIR = None  # Path("")
    DDIR = None  # Path("")

    if WDIR is None or DDIR is None:
        if os_name == "Windows":
            WDIR = Path(r"C:\Users\lukas\Documents\git\compla")
            DDIR = Path(r"C:\Users\lukas\Documents\git\compla\data")
        elif os_name == "Linux":
            WDIR = Path(r"/home/lukasw/Documents/git/compla")
            DDIR = Path(r"/dump/lukasw/data")
        else:
            print("Assign WDIR and DDIR")
            raise NotImplementedError

    return WDIR, DDIR





if __name__ == "__main__":

    get_wdir_ddir()