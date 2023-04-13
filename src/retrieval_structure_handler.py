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

WDIR = Path().cwd().parent

PLANET_DB_PATH = str(WDIR / "data/planet_database_composite.csv")
MOLECULE_PATH = str(WDIR / "data/molecule_db.json")

OPACITY_PATH = str(WDIR / "data/Input/xsec/xsec_sampled_R15000_0.3-15")
CIA_PATH = str(WDIR / "data/Input/cia/hitran")
KTABLE_PATH = str(WDIR / "data/Input/ktables/R100")

SPECTRA_BE_PATH = str(WDIR / "data/SpectraBE")
SPECTRA_LW_PATH = str(WDIR / "data/taurex_lightcurves_LW")

def parse_file_names(file_names):
    parsed_data = []

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
                'spectral_element': properties[1]
            }
        else:
            print(f"Invalid file format: {file_name}")
            continue

        parsed_data.append(file_data)

    return parsed_data

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
    not_common_keys = [key for key in merged_dict if key not in keys_in_order]

    if isinstance(merged_dict['planet_name'], list):
        raise ValueError("Planet name must be common among all entries")

    common_part = '_'.join([merged_dict[key] for key in keys_in_order if not isinstance(merged_dict[key], list)])
    not_common_part = '_'.join([f"{key}(" + '_'.join([f"{value}" for value in merged_dict[key]]) + ")" for key in keys_in_order + not_common_keys if isinstance(merged_dict[key], list)])

    filename = f"{common_part}_{not_common_part}"
    return filename


def create_path(merged_dict):
    base_path = Path("/data/retrievals")
    keys_in_order = ['planet_name', 'facility', 'instrument', 'spectral_element']

    common_keys = []
    for key in keys_in_order:
        if isinstance(merged_dict[key], list):
            break
        common_keys.append(key)

    new_directory = create_filename(merged_dict)
    path_parts = [base_path] + [merged_dict[key] for key in common_keys] + [new_directory]
    path = Path(*path_parts)

    return path

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

    merged = merge_dicts(proplist)
    path = create_path(merged)
    filename = create_filename(merged)

    print(merged)
    print(path)
    print(filename)