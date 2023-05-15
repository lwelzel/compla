import numpy as np
import sys
import re
import os
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, Callable
from tqdm import tqdm
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from configobj import ConfigObj, Section
import shutil
from tempfile import NamedTemporaryFile
from mpi4py import MPI  # import the 'MPI' module

from utils import get_wdir_ddir


WDIR, DDIR = get_wdir_ddir()

PLANET_DB_PATH = str(WDIR / "data/planet_database_composite.csv")
MOLECULE_PATH = str(WDIR / "data/molecule_db.json")

OPACITY_PATH = str(WDIR / "data/Input/xsec/xsec_sampled_R15000_0.3-15")
CIA_PATH = str(WDIR / "data/Input/cia/hitran")
KTABLE_PATH = str(WDIR / "data/Input/ktables/R100")

SPECTRA_BE_PATH = str(WDIR / "data/SpectraBE")
SPECTRA_LW_PATH = str(WDIR / "data/taurex_lightcurves_LW")


def parse_file_names(file_names):
    parsed_data = []

    if isinstance(file_names, str):
        file_names = [file_names]

    file_names = [Path(filename).name for filename in file_names]

    synthetic = False
    if np.any(["synthetic" in filename for filename in file_names]):
        synthetic = True

    file_names = [filename.replace("synthetic_", "") for filename in file_names]
    file_names = [re.sub(r'_transmission_spectrum_(\d+)\.txt', r',TM\1', filename) for filename in file_names]

    for file_name in file_names:
        file_name = file_name.rstrip('.txt')

        properties = file_name.split('_')
        prop_count = len(properties)

        # Check if it is the first or second kind of file
        if prop_count in [6, 7]:
            file_data = {
                'planet_name': properties[0],
                'facility': properties[1],
                'instrument': properties[2],
                'spectral_element': properties[3],
                'aperture': properties[4],
                'source': properties[5],
                'bandwidth': False if prop_count == 6 else True
            }
        elif prop_count == 2:
            file_data = {
                'planet_name': properties[0],
                'facility': "HST",
                'instrument': "WFC3",
                'source': "Edwards+2022",
                'spectral_element': properties[1]
            }
        else:
            print(f"Invalid file format: {file_name}")
            continue

        for s in ["-b_", "-c_", "-d_", "-e_", "-f_", ]:
            file_data['planet_name'] = file_data['planet_name'].replace(s, s.replace("-", ""))

        parsed_data.append(file_data)

    return parsed_data, synthetic

def merge_dicts(dict_list):
    merged_dict = {}

    all_keys = set().union(*(d.keys() for d in dict_list))

    for key in all_keys:
        values = [d.get(key, "NA") for d in dict_list]
        unique_values = list(set(values))

        if len(unique_values) == 1:
            merged_dict[key] = unique_values[0]
        else:
            merged_dict[key] = unique_values

    return merged_dict

def create_filename(merged_dict):
    keys_in_order = ['planet_name', 'facility', 'instrument', 'spectral_element']
    ignore_keys = ["bandwidth"]
    not_common_keys_in_order = [
        "aperture",
        "source",
    ]
    not_common_keys = [key for key in not_common_keys_in_order if key in merged_dict.keys() and key not in keys_in_order and key not in ignore_keys]


    if isinstance(merged_dict['planet_name'], list):
        raise ValueError("Planet name must be common among all entries")

    common_part = '_'.join([merged_dict[key] for key in keys_in_order
                            if (not isinstance(merged_dict[key], list) or len(merged_dict[key]) == 1)])
    not_common_part = []
    for key in keys_in_order + not_common_keys:
        if isinstance(merged_dict[key], list):
            not_common_part.append(f"{key}(" + '_'.join(sorted([f"{value}" for value in merged_dict[key]])) + ")")
        elif key in not_common_keys and isinstance(merged_dict[key], str):
            not_common_part.append(merged_dict[key])

    not_common_part = "_".join(not_common_part)

    filename = f"{common_part}_{not_common_part}"
    return filename


def create_path(merged_dict, synthetic=False):
    if synthetic:
        base_path = DDIR / Path("SYN")
    else:
        base_path = DDIR / Path("OBS")

    keys_in_order = ['planet_name', ]  # 'facility', 'instrument', 'spectral_element']

    common_keys = []
    for key in keys_in_order:
        if isinstance(merged_dict[key], list):
            break
        common_keys.append(key)

    new_directory = create_filename(merged_dict)
    path_parts = [base_path] + [merged_dict[key] for key in common_keys] + [new_directory]
    path = Path(*path_parts)

    return str(path)[1:]


def get_path_filename(file_names, synthetic=False):
    dicts, _synthetic = parse_file_names(file_names)
    show_new_path = False
    if _synthetic != synthetic and synthetic is False:
        show_new_path = True
        warnings.warn(f"Path parser was set to write non-synthetic files but input looks synthetic. Setting synthetic to True.\n"
                      f"\tInput files: {file_names}"
                      )
        synthetic = True

    merged = merge_dicts(dicts)
    path = create_path(merged, synthetic=synthetic)
    if show_new_path:
        print(f"Setting new path to -> {path}")
    filename = create_filename(merged)

    pattern = r'_\d+$'
    if isinstance(path, Path):
        path = str(path)
        path = re.sub(pattern, '', path)
        path = Path(path)
    else:
        path = re.sub(pattern, '', path)

    return path, filename


def get_retrieval_path():
    pass



if __name__ == "__main__":
    file_data1 = {
        'planet_name': "P",
        'facility': "HST",
        'instrument': "STIS",
        'spectral_element': "G141",
        'aperture': "APP1",
        'source': "S1",
        'bandwidth': "True"
    }

    file_data2 = {
        'planet_name': "P",
        'facility': "HST",
        'instrument': "STIS",
        'spectral_element': "G141",
    }

    file_data3 = {
        'planet_name': "P",
        'facility': "JWST",
        'instrument': "WFC3",
        'spectral_element': "G141",
        'aperture': "APP2",
        'source': "S2",
        'bandwidth': "False"
    }

    proplist = [file_data1, file_data2, file_data3]

    file_names = [
        str(WDIR / "data/taurex_lightcurves_LW" / "HAT-P-1-b_HST_STIS_G430L_52X2_Nikolov+2014.txt"),
        str(WDIR / "data/taurex_lightcurves_LW" / "HAT-P-1-b_HST_STIS_G430L_52X2_Sing+2016.txt"),
        str(WDIR / "data/SpectraBE" / "HAT-P-1b_G141.txt")
    ]

    path, filename = get_path_filename(file_names, synthetic=True)

    print(path)
    print(filename)
