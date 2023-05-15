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
from src.retrieval_structure_handler import get_path_filename
from utils import get_wdir_ddir
from mpi4py import MPI  # import the 'MPI' module

WDIR, DDIR = get_wdir_ddir()

PLANET_DB_PATH = str(DDIR / "planet_database_composite.csv")
MOLECULE_PATH = str(DDIR / "molecule_db.json")

OPACITY_PATH = str(DDIR / "Input/xsec/xsec_sampled_R15000_0.3-15")
CIA_PATH = str(DDIR / "Input/cia/HITRAN")
KTABLE_PATH = str(DDIR / "Input/ktables/R100")

SPECTRA_BE_PATH = str(DDIR / "SpectraBE")
SPECTRA_LW_PATH = str(DDIR / "taurex_lightcurves_LW")


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def read_csv_comments(filename: str) -> list:
    lines = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row) > 0 and row[0].startswith("# COLUMN"):
                lines.append(row[0])
    return lines


def process_lines(file_path) -> Dict:
    lines = read_csv_comments(file_path)

    result = {}
    for line in lines:
        key, value = line.replace("# COLUMN", "").split(":", 1)
        result[key.strip()] = value.strip()
    return result


# TODO: own file
@dataclass
class AliasedDict(dict):
    aliases: dict

    def __init__(self, data: dict, aliases: dict):
        self.data = data
        super().__init__(data)
        self.aliases = aliases

    def __getitem__(self, key):
        if key in self.aliases:
            key = self.aliases[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self.aliases:
            key = self.aliases[key]
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        if key in self.aliases:
            key = self.aliases[key]
        return super().__delitem__(key)

    def add_alias(self, key, alias):
        self.aliases[alias] = key


# TODO: own file
class TaurexConfigObj(ConfigObj):
    def _write_line(self, indent_string, entry, this_entry, comment):

        try:
            if re.search(":fit", entry):
                indent_string = os.linesep + indent_string
        except TypeError:
            print(entry)
        finally:
            return super()._write_line(indent_string, entry, this_entry, comment)

    def _write_marker(self, indent_string, depth, entry, comment):
        """Write a section marker line"""
        entry_str = self._decode_element(entry)
        title = self._quote(entry_str, multiline=False)
        if entry_str and title[0] in '\'"' and title[1:-1] == entry_str:
            # titles are in '[]' already, so quoting for contained quotes is not necessary (#74)
            title = entry_str
        return '%s%s%s%s%s' % (os.linesep + indent_string,
                               '[' * depth,
                               title,
                               ']' * depth,
                               self._decode_element(comment))


# TODO: instead of defining import from other file
def get_target_data(planet_name):
    exoplanet_database = pd.read_csv(PLANET_DB_PATH, comment="#", index_col=0)

    target_data = exoplanet_database.loc[exoplanet_database["pl_name"] == planet_name].to_dict(orient="records")[0]

    col_names = process_lines(PLANET_DB_PATH)
    col_names = {v: k for k, v in col_names.items()}

    target = AliasedDict(target_data, aliases=col_names)

    return target

def make_Global_dict(settings=None, **kwargs):
    global_dict = {
        'xsec_path': str(OPACITY_PATH),
        "cia_path": str(CIA_PATH),
        "ktable_path": str(KTABLE_PATH),
        "opacity_method": "xsec",
        "mpi_use_shared": True,
    }

    return {"Global": global_dict}


def make_Obs_ObsFit_dict(path_list=None, fit_list=None, fitting_bounds=None, fitting_mode="linear", settings=None,
                         **kwargs):
    if path_list is None:
        raise ValueError

    if isinstance(path_list, list) and not len(path_list) == 1:
        observation_dict = {
            'observation': 'spectra_w_offsets',
            'path_spectra': [str(p) for p in path_list],
            'offsets': ["0.0"] * len(path_list),
            'slopes': ["0.0"] * len(path_list),
        }

        if fitting_bounds is None:
            fitting_bounds = ['-1e-3', '1e-3']

        if fit_list is None:
            fit_list = list(np.full_like(path_list, fill_value=True, dtype=bool))
            fit_list[0] = False

        if isinstance(fitting_bounds[0], (str, float)) and isinstance(fitting_mode, str):
            fitting_dict = {}
            for i, (__, fit) in enumerate(zip(path_list, fit_list)):
                obs_fit_dict = {
                    f'Offset_{i + 1}:fit': fit,
                    f'Offset_{i + 1}:prior': f"Uniform(bounds=({fitting_bounds[0]}, {fitting_bounds[1]}))",
                    f'Offset_{i + 1}:bounds': fitting_bounds,
                    f'Offset_{i + 1}:mode': "linear",
                    f'Slope_{i + 1}:fit': False,
                    f'Slope_{i + 1}:prior': f"Uniform(bounds=({fitting_bounds[0]}, {fitting_bounds[1]}))",
                    f'Slope_{i + 1}:bounds': fitting_bounds,
                    f'Slope_{i + 1}:mode': "linear",
                }

                fitting_dict = {**fitting_dict, **obs_fit_dict}
        elif fitting_mode != "linear":
            raise NotImplementedError(f"Only linear fitting is implemented. Requested: {fitting_mode}")
        else:
            raise NotImplementedError("Only same options for each observation fit is supported.")

        out_dict = {
            'Observation': observation_dict,
            'Fitting': fitting_dict,
        }
        return out_dict

    elif isinstance(path_list, str) or len(path_list) == 1:
        if isinstance(path_list, list):
            path_list = path_list[0]

        observation_dict = {
            'observed_spectrum': path_list,
        }
        out_dict = {
            'Observation': observation_dict,
            'Fitting': {},
        }
        return out_dict
    else:
        raise NotImplementedError(f"Path list not accepted. Input path_list: {path_list}")


def make_Chem_dict(molecule_dp_path=MOLECULE_PATH, which_molecules=None,
                   fit_list=None, fit_bounds=None, fit_modes=None,
                   settings=None, **kwargs):
    # TODO: use types as type hinting?
    types = {
        'ConstantGas': "constant",
        "TwoPointGas": "twopoint",
        'TwoLayerGas': "twolayer",
        'HydrogenIon': None,
    }

    # TODO: unpack, into which gasses, k, v
    gas_para_db = read_json_file(MOLECULE_PATH)
    gas_para_db_dict = {gas["molecule"]: gas for gas in gas_para_db}

    gases_dict = {}
    fitting_dict = {}

    if which_molecules is None:
        which_molecules = gas_para_db_dict.keys()

    if fit_bounds is None:
        fit_bounds = [[1e-16, 1e-1] for __ in which_molecules]

    if fit_modes is None:
        fit_modes = ["LogUniform(lin_bounds=({}, {}))"] * len(which_molecules)

    if fit_list is None:
        fit_list = list(np.full(len(which_molecules),
                                fill_value=True,
                                dtype=bool))

    assert len(np.unique([len(l) for l in [which_molecules, fit_list, fit_bounds, fit_modes]])) == 1

    # TODO: add gas_type as attr to json separate from type? for .par files
    for i, (gas, fit_, fit_bounds, fit_mode) in enumerate(zip(which_molecules, fit_list, fit_bounds, fit_modes)):
        try:
            _gas = gas_para_db_dict[gas]
        except KeyError:
            raise KeyError(f"Gas {gas} was not found in gas DB. Possible keys: {gas_para_db.keys()}")
        if types[_gas["type"]] == "constant":
            gas_dict = {_gas["molecule"]:
                {
                    "gas_type": types[_gas["type"]],
                    "mix_ratio": 10 ** _gas["abundance"],
                }
            }

        else:
            raise KeyError(
                f"Gas type {types[_gas['type']]} not implemented. Gas {_gas['molecule']} requested {types[_gas['type']]}")

        # check for violated bounds
        if 10 ** _gas["abundance"] < fit_bounds[0]:
            fit_bounds[0] = 0.1 * 10 ** _gas["abundance"]
        if 10 ** _gas["abundance"] > fit_bounds[1]:
            fit_bounds[1] = np.minimum(2. * 10 ** _gas["abundance"], 0.9)

        fit_dict = {
            f'{_gas["molecule"]}:fit': fit_,
            f'{_gas["molecule"]}:prior': fit_mode.format(fit_bounds[0], fit_bounds[1]),
            f'{_gas["molecule"]}:bounds': fit_bounds,
            f'{_gas["molecule"]}:mode': "log" if "Log" in fit_mode else "linear",
        }

        gases_dict = {**gases_dict, **gas_dict}
        fitting_dict = {**fitting_dict, **fit_dict}

    chemistry_dict = {
        "chemistry_type": "free",  # note: free is the same as taurex?
        "fill_gases": ["H2", "He"],
        "ratio": 0.17,
        **gases_dict,
    }

    return {"Chemistry": chemistry_dict, "Fitting": fitting_dict}


def make_FastChem_dict(target, molecule_dp_path=MOLECULE_PATH, which_molecules=None,
                       which_ratios=None, which_ratios_val=None,
                       fit_list=None, fit_bounds=None, fit_modes=None,
                       settings=None, **kwargs):


    # TODO: unpack, into which gasses, k, v
    gas_para_db = read_json_file(MOLECULE_PATH)

    if which_molecules is None:
        which_molecules = ["H", "He", "C", "N", "O",'N','K','e-']

    if which_ratios is None:
        which_ratios = ["C",  # _to_O
                        # "N",  # _to_O
                        ]
        if len(which_ratios) == 1:
            which_ratios = which_ratios[0]
    if which_ratios_val is None:
        which_ratios_val = [0.5,
                            # 1e-3,
                            ]
        if len(which_ratios_val) == 1:
            which_ratios_val = which_ratios_val[0]
        # which_ratios_val = [1. for __ in which_ratios]


    if fit_bounds is None:
        fit_bounds = [[1e-2, 2.] for __ in which_ratios]

    if fit_modes is None:
        fit_modes = ["LogUniform(lin_bounds=({}, {}))"] * len(which_ratios)

    if fit_list is None:
        fit_list = list(np.full(len(which_ratios),
                                fill_value=True,
                                dtype=bool))

    # solar_metallicity = 0.0196
    # try:
    #     metallicity = float(target.data['st_metratio']) / solar_metallicity
    # except ValueError:
    #     metallicity = 10 ** float(target.data['st_met']) * solar_metallicity

    chemistry_dict = {
        "chemistry_type": "fastchem",
        "metallicity": 1.,  # Metallicity relative to initial abundance
        # "selected_elements": which_molecules,  # Do not select any -> defaults to all
        "ratio_elements": which_ratios,
        "ratios_to_O": which_ratios_val,
        "with_ions": True,
    }

    fitting_dict = {
        "metallicity:fit": True,
        "metallicity:prior": "LogUniform(bounds=(-2, 2))",
        "metallicity:bounds": [0.01, 100],
        "metallicity:mode": "log",
    }

    # assert len(np.unique([len(l) for l in [which_ratios, fit_list, fit_bounds, fit_modes]])) == 1

    # TODO: add gas_type as attr to json separate from type? for .par files
    for i, (ratio, fit_, fit_bounds, fit_mode) in enumerate(zip(which_ratios, fit_list, fit_bounds, fit_modes)):
        fit_dict = {
            f'{ratio}_O_ratio:fit': fit_,
            f'{ratio}_O_ratio:prior': fit_mode.format(fit_bounds[0], fit_bounds[1]),
            f'{ratio}_O_ratio:bounds': fit_bounds,
            f'{ratio}_O_ratio:mode': "log" if "Log" in fit_mode else "linear",
        }

        fitting_dict = {**fitting_dict, **fit_dict}

    return {"Chemistry": chemistry_dict, "Fitting": fitting_dict}

def make_ACE_dict(target, molecule_dp_path=MOLECULE_PATH, which_molecules=None,
                       which_ratios=None, which_ratios_val=None,
                       fit_list=None, fit_bounds=None, fit_modes=None,
                       settings=None, **kwargs):

    # solar_metallicity = 0.0196
    # try:
    #     metallicity = float(target.data['st_metratio']) / solar_metallicity
    # except ValueError:
    #     metallicity = 10 ** float(target.data['st_met']) * solar_metallicity

    chemistry_dict = {
        "chemistry_type": "ace",
        "metallicity": 1.,  # Metallicity relative to initial abundance
        "co_ratio": 0.5,
    }

    # TODO: check if properly implemented
    active_molecules = ['CH3COOOH', 'C4H9O', 'C3H7O', 'NO3', 'CH3COOO', 'C2H5OO', 'C2H4OOH', 'HONO2', 'C2H5OOH',
                        'CH3ONO', 'C3H8CO', 'CH3NO2', '1C4H9', '2C4H9', 'C4H10', 'C3H7OH', 'CH3OO', 'C4H8Y', 'CH3OOH',
                        'HNO2', 'CH3OCO', 'C2H5CHO', 'C2H6CO', 'C2H5O', 'CH3NO', '2C2H4OH', 'NO2', '2C3H7', '1C3H7',
                        '1C2H4OH', 'HONO', 'C3H8', 'HCNN', 'cC2H4O', 'HCNO', 'C2H5OH', 'N2O', 'C2H3CHOZ', 'OOH',
                        'CH2CHO', 'H2O2', 'CH3CO', 'NCO', 'CH3O', 'O2', 'CH3CHO', 'HNO', 'C', 'CHCO', 'CO2H', 'HOCN',
                        'C2H5', 'C2H', 'CH2OH', 'CH', 'C2H6', 'C2H3', 'CH2CO', 'NNH', 'H2CN', 'CH3OH', 'N4S', 'N2D',
                        'CN', '1CH2', 'HNCO', 'NO', 'O3P', 'O1D', 'C2H4', 'NH', '3CH2', 'HCO', 'C2H2', 'H2CO', 'NH2',
                        'CO2', 'OH', 'CH3', 'HCN', 'NH3', 'CH4', 'N2', 'CO', 'H2O', 'H', 'He', 'H2', 'N2O4', 'N2O3',
                        'N2H2', 'N2H3', 'N2H4', 'HNNO', 'HNOH', 'HNO3', 'NH2OH', 'H2NO', 'CNN', 'H2CNO', 'C2N2', 'HCNH',
                        'HNC', 'HON', 'NCN']

    ace_metallicity_bounds = [0.01, 10.]
    ace_metallicity_prior = "Uniform(bounds=({}, {}))"
    C_O_ratio_bounds = [1e-4, 10.]
    C_O_ratio_prior = "Uniform(bounds=({}, {}))"


    fitting_dict = {
        "ace_metallicity:fit": True,
        "ace_metallicity:prior": ace_metallicity_prior.format(*ace_metallicity_bounds),
        f'ace_metallicity:bounds': ace_metallicity_bounds,
        f'ace_metallicity:mode': "log" if "Log" in ace_metallicity_prior else "linear",

        "C_O_ratio:fit": True,
        "C_O_ratio:prior": C_O_ratio_prior.format(*C_O_ratio_bounds),
        f'C_O_ratio:bounds': C_O_ratio_bounds,
        f'C_O_ratio:mode': "log" if "Log" in C_O_ratio_prior else "linear",
    }

    return {"Chemistry": chemistry_dict, "Fitting": fitting_dict}


def make_Temp_dict(target=None, settings=None, which=None, **kwargs):
    if target is None:
        raise ValueError

    if settings is None or "Temperature" not in settings:
        settings = {}

    # TODO: default is isothermal. Check re npoint or Guillot2016/2018?
    default_iso_dict = {
        "profile_type": "isothermal",
        "T": target["Equilibrium Temperature [K]"],
    }

    default_guillot_dict = {
        "profile_type": "guillot",
        "T_irr": target["Equilibrium Temperature [K]"],
        # rest is defaults for now
    }

    default_npoint_dict = {
        "profile_type": "npoint",
        "T_top": 0.8 * target["Equilibrium Temperature [K]"],
        "T_surface": target["Equilibrium Temperature [K]"],
        "P_top": 1.0e-6,
        "P_surface": 1.0e6,
        "temperature_points": [0.99 * target["Equilibrium Temperature [K]"],
                               0.9 * target["Equilibrium Temperature [K]"]],
        "pressure_points": [1.0e-3, 1.0e3],
        # rest is defaults for now
    }

    if which is None:
        default_dict = default_iso_dict
    elif which == "isothermal":
        default_dict = default_iso_dict
    elif which == "guillot":
        default_dict = default_guillot_dict
    else:
        raise NotImplementedError

    # TODO: issues because errors are thrown if unexpected keys in Temperature config
    # overwrite partial if type key is same, overwrite fully if mismatch
    return {"Temperature": {**default_dict,
                            **settings.get("Temperature", settings)}}


def make_Press_dict(settings=None, **kwargs):
    if settings is None or "Pressure" not in settings:
        settings = {"profile_type": "simple"}

    return {"Pressure": settings.get('Pressure', settings)}


def make_Planet_dict(target=None, settings=None, **kwargs):
    if target is None:
        raise ValueError
    if settings is None or "Planet" not in settings:
        settings = {}

    keys_header = [
        "planet_mass",
        "planet_radius",
        "planet_distance",
        "impact_param",
        "orbital_period",
        "albedo",
        "transit_time",
    ]

    keys_db = [
        "Planet Mass or Mass*sin(i) [Jupiter Mass]",
        "Planet Radius [Jupiter Radius]",
        'Orbit Semi-Major Axis [au])',
        'Impact Parameter',
        'Orbital Period [days]',
        "NO MATCHING KEY",
        'Transit Duration [hours]',
    ]

    planet_dict = {"planet_type": "simple"}

    for k1, k2 in zip(keys_header, keys_db):
        try:
            v = target[k2]
            assert isinstance(v, float) and np.isfinite(v)
            planet_dict[k1] = v
        except (KeyError, AssertionError):
            pass

    return {"Planet": {**planet_dict, **settings.get('Planet', settings)}}


def make_Star_dict(target=None, settings=None, **kwargs):
    if target is None:
        raise ValueError
    # TODO: PHOENIX library normally used?
    if settings is None or "Star" not in settings:
        settings = {}

    keys_header = [
        "temperature",
        "radius",
        "mass",
        "distance",
        "metallicity",
        "magnitudeK",
    ]

    keys_db = [
        'Stellar Effective Temperature [K]',
        'Stellar Radius [Solar Radius]',
        'Stellar Mass [Solar mass]',
        'Distance [pc]',
        'Stellar Metallicity Ratio',
        'Ks (2MASS) Magnitude',
    ]

    solar_metallicity = 0.0196

    star_dict = {"star_type": "blackbody"}

    for k1, k2 in zip(keys_header, keys_db):
        try:
            v = target[k2]
            assert isinstance(v, float) and np.isfinite(v)
            if k1 == "metallicity" or k2 == 'Stellar Metallicity Ratio':
                star_dict[k1] = v / solar_metallicity
                continue
            star_dict[k1] = v
        except (KeyError, AssertionError):
            pass

    return {"Star": {**star_dict, **settings}}


def make_FW_dict(target=None, settings=None, **kwargs):
    if target is None:
        raise ValueError
    if settings is None or "Model" not in settings:
        settings = {}

    model_dict = {
        "model_type": "transmission",
        "Absorption": {},
        "CIA": {"cia_pairs": ['H2-H2', 'H2-He']},
        "Rayleigh": {},
        # "SimpleClouds": {"clouds_pressure": 0.1}, # OR:
        "ThickClouds": {"clouds_pressure": 1e3},
        # "HydrogenIon": {}, TODO: manually enable for fastchem
        # "LeeMie": {},  # OR:
        # 'BHMie': {},  # OR:
        # "FlatMie": {},
    }

    return {"Model": {**model_dict, **settings.get("Model", settings)}}


def make_Bin_dict(settings=None, **kwargs):
    if settings is None or "Binning" not in settings:
        settings = {"bin_type": "observed"}

    return {"Binning": settings.get("Binning", settings)}

def make_Fit_dict(tm=None, target=None, which=None, settings=None, default=False, **kwargs):
    if target is None:
        raise ValueError

    if (tm is None and which is None) and not default:
        raise ValueError(f"tm is not defined yet, cannot set fitting dict without 'which': {which}")

    if settings is None:
        settings = {}

    if tm is not None:
        possible_fit_para = [k for k, v in tm.fittingParameters.items()]
        if which is None:
            which = possible_fit_para
    elif tm is None and default:
        which = [
            "planet_radius",
            "T",
        ]

    default_fit_para = {
        "planet_mass": {
            "fit": False,
            "bounds": [np.nan, np.nan],
            "prior": "Uniform(bounds=({}, {}))",
            "value": target["Planet Mass or Mass*sin(i) [Jupiter Mass]"]
        },
        "planet_radius": {
            "fit": True,
            "bounds": [0.5 * target["Planet Radius [Jupiter Radius]"],
                       1.5 * target["Planet Radius [Jupiter Radius]"]],
            "prior": "Uniform(bounds=({}, {}))",
            "value": target["Planet Radius [Jupiter Radius]"]
        },
        # "planet_distance" : {
        #     "fit": True,
        #     "bounds": [],
        #     "prior": "Uniform(bounds=({}, {}))"
        # },
        # "planet_sma" : {
        #     "fit": True,
        #     "bounds": [],
        #     "prior": "Uniform(bounds=({}, {}))"
        # },
        "T": {
            "fit": True,
            "bounds": [np.maximum(0.33 * target["Equilibrium Temperature [K]"], 300.),
                       np.maximum(2. * target["Equilibrium Temperature [K]"], 2500.)],
            "prior": "Uniform(bounds=({}, {}))",
            "value": target["Equilibrium Temperature [K]"]
        },
        "T_irr": {
            "fit": True,
            "bounds": [np.maximum(0.33 * target["Equilibrium Temperature [K]"], 300.),
                       np.maximum(2. * target["Equilibrium Temperature [K]"], 2500.)],
            "prior": "Uniform(bounds=({}, {}))",
            "value": target["Equilibrium Temperature [K]"]
        },
        "atm_min_pressure": {
            "fit": False,
            "bounds": [1e-8, 1e-2],
            "prior": "LogUniform(lin_bounds=({}, {}))",
            "value": 1e-4,
        },
        "atm_max_pressure": {
            "fit": False,
            "bounds": [1e4, 1e8],
            "prior": "LogUniform(lin_bounds=({}, {}))",
            "value": 1e6
        },
        "clouds_pressure": {
            "fit": True,
            "bounds": [1e-2, 1e5],
            "prior": "LogUniform(lin_bounds=({}, {}))",
            "value": 0.1
        },
        "nlayers": {
            "fit": False,
            "bounds": [np.nan, np.nan],
            "prior": "Uniform(bounds=({}, {}))",
            "value": 100
        },
    }

    fit_dict = {}

    for para in which:
        try:
            assert para in possible_fit_para.keys(), f"Cant fit {para} because it is not in the possible fitting parameters: {possible_fit_para}"
        except UnboundLocalError as e:
            if default:
                pass
            else:
                raise e
        assert para in default_fit_para.keys(), f"Cant fit {para} because it is not in the default fitting parameter: {default_fit_para.keys()}. Ignore this if the fit is set elsewhere."
        try:

            value = default_fit_para[para]["value"]

            # check for violated bounds
            if value < default_fit_para[para]["bounds"][0]:
                default_fit_para[para]["bounds"][0] = 0.5 * value
            if value > default_fit_para[para]["bounds"][1]:
                default_fit_para[para]["bounds"][1] = 2 * value

            para_fit_dict = {
                f'{para}:fit': default_fit_para[para]["fit"],
                f'{para}:prior': default_fit_para[para]["prior"].format(*default_fit_para[para]["bounds"]),
                f'{para}:bounds': default_fit_para[para]["bounds"],
                f'{para}:mode': "log" if "Log" in default_fit_para[para]["prior"] else "linear",
            }

            fit_dict = {**fit_dict, **para_fit_dict}

        except KeyError:
            pass
        except AssertionError as e:
            print(e)
            pass

    return {"Fitting": {**fit_dict, **settings}}


def make_Opt_dict(settings=None, path=None, filename=None, **kwargs):
    # if settings is None or "Optimizer" not in settings:
    #     settings = {"optimizer": "nestle"}
    # if path is not None and filename is not None and settings["optimizer"] == "ultranest":
    #     settings = {**settings, **{"log_dir": str(Path(path) / Path(filename).stem)}}

    # settings = {**settings,
    #             **{
    #                 "optimizer": "ultranest",
    #                 "num_live_points": 400,
    #                 'dlogz': 0.5,
    #                 "dkl": 0.5,
    #                 'max_num_improvement_loops': 0,
    #                 "stepsampler": "RegionBallSliceSampler",
    #                 "log_dir": str(Path(path) / Path(filename).stem),
    #             }}

    settings = {**settings,
                **{
                    "optimizer": "multinest",
                    "num_live_points": 1000,
                    'evidence_tolerance': 0.1,  # set to 6-8 for fast convergence testing
                    "max_iterations": 0,
                    "multi_nest_path": str(Path(path)),  # not supported?
                }}

    return {"Optimizer": settings}


def make_Inst_dict(settings=None, **kwargs):
    if settings is None or "Instrument" not in settings:
        settings = {
            "instrument": "snr",
                    "SNR": 100000,
                    "num_obs": 1,
                    }

    return {"Instrument": settings}

def make_bounds_from_derived_para(para, *args, interval=None, mode=None, fit=True, **kwargs):
    if interval is None:
        interval = np.array([0.1, 10.])
    if mode is None:
        mode = "linear"  # TODO: check with docs if custom priors instead of bounds

    out = {
        f"{para['name']}:fit": fit,
        f"{para['name']}:bounds": list(np.sort(interval * para["value"])),
        f"{para['name']}:mode": mode,
    }

    return {"Fitting": out}


def make_Derive_dict():
    derive_dict = {
        "mu:compute": True,
        "logg:compute": True,
        "avg_T:compute": True,
        "C_O_ratio:compute": True,
        "He_H_ratio:compute": True,
    }

    return {"Derive": derive_dict}


def unpack_dicts(dicts, ignore_keys=None, ignore_overwrite=False):
    if ignore_keys is None:
        ignore_keys = []

    def _unpack_dict(d, out_dict):
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in out_dict:
                    out_dict[k] = {}
                _unpack_dict(v, out_dict[k])
            elif k in out_dict and (k not in ignore_keys or ignore_overwrite):
                raise ValueError(f"Overwriting non-dict value for key '{k}'")
            else:
                out_dict[k] = v
        return out_dict

    out_dict = {}
    for d in dicts:
        out_dict = _unpack_dict(d, out_dict)
    return out_dict


def _write_par_file(path_list, target, tm=None, settings=None, which_molecules=None, comments=None,
                    path=None, filename=None, fastchem=False, ace=False, synthetic=False, ):
    if isinstance(target, str):
        target = get_target_data(target)

    if settings is None:
        settings = {}

    if path is None or filename is None:
        time = f'_time-{datetime.now().isoformat(sep="-", timespec="seconds").replace(":", "-")}'

        path, filename = get_path_filename(path_list, synthetic=synthetic, timestamp=True)

        os.makedirs(str(path), exist_ok=True)
        filename = str(filename) + ".par"
        filename = "parfile.par"

    global_dict = make_Global_dict(settings=settings)
    obs_dict = make_Obs_ObsFit_dict(path_list=path_list, settings=settings)
    if fastchem:
        chem_dict = make_FastChem_dict(target=target, which_molecules=which_molecules, settings=settings)
    elif ace:
        chem_dict = make_ACE_dict(target=target, which_molecules=which_molecules, settings=settings)
    else:
        chem_dict = make_Chem_dict(which_molecules=which_molecules, settings=settings)
    temp_dict = make_Temp_dict(target=target, settings=settings)
    press_dict = make_Press_dict(settings=settings)
    planet_dict = make_Planet_dict(target=target, settings=settings)
    star_dict = make_Star_dict(target=target, settings=settings)
    fw_dict = make_FW_dict(target=target, settings=settings)
    fit_dict = make_Fit_dict(tm=tm, target=target, default=True, settings=settings)
    opt_dict = make_Opt_dict(settings=settings, path=path, filename=filename)

    dict_list = [
        global_dict,
        obs_dict,
        chem_dict,
        temp_dict,
        press_dict,
        planet_dict,
        star_dict,
        fw_dict,
        fit_dict,
        opt_dict,
    ]

    if synthetic:
        bin_dict = make_Bin_dict(settings=settings)
        inst_dict = make_Inst_dict(settings=settings)

        dict_list.append(bin_dict)
        dict_list.append(inst_dict)

    par_dict = unpack_dicts(dict_list)

    config = TaurexConfigObj(par_dict)

    config.initial_comment.append(
        f'# Created: {datetime.now().isoformat(sep="-", timespec="seconds").replace(":", "-")}')
    if comments is not None and isinstance(comments, (list, np.ndarray)):
        for c in comments:
            assert isinstance(c, str)
            config.initial_comment.append("# " + c)

    # if path is None:
    #     dir_name = "DEFAULT"
    #     path = str(WDIR / "data/retrievals" / dir_name)
    #     os.makedirs(path, exist_ok=True)
    #
    # if filename is None:
    #     filename = "default.par"

    config_path = Path(path) / filename

    config.filename = str(Path(path) / filename)

    # Save the file to a temporary location if it exists
    temp_file = None
    if config_path.exists():
        temp_file = NamedTemporaryFile(delete=False)
        shutil.copy(str(config_path), temp_file.name)

    # Try writing the config to the file
    try:
        config.write()
    except Exception as e:
        # If writing fails, restore the file from the temporary location
        if temp_file:
            shutil.copy(temp_file.name, str(config_path))
            temp_file.close()
            os.unlink(temp_file.name)
        raise e

    return config_path / config.filename


def write_par_file(path_list, target, tm=None, settings=None, which_molecules=None, comments=None,
                   path=None, filename=None, fastchem=False, ace=False, synthetic=False, ):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        path = _write_par_file(path_list=path_list, target=target, tm=tm, settings=settings, which_molecules=which_molecules, comments=comments,
                               path=path, filename=filename, fastchem=fastchem, ace=ace, synthetic=synthetic, )

    comm.Barrier()
    path = comm.bcast(path, root=0)

    return path


if __name__ == "__main__":
    path_list = [
        str(DDIR / "taurex_lightcurves_LW" / "WASP-39-b_HST_STIS_G430L_52X2_Sing+2016.txt"),
        str(DDIR / "taurex_lightcurves_LW" / "WASP-39-b_HST_WFC3_G141_GRISM256_Wakeford+2018.txt"),
    ]

    # path_list = [
    #     # str(WDIR / "data/synthetic_spectra/HAT-P-1b" / "synthetic_HAT-P-1-b_HST_STIS_G430L_52X2_Nikolov+2014_transmission_spectrum_0.txt"),
    #     # str(WDIR / "data/synthetic_spectra/HAT-P-1b" / "synthetic_HAT-P-1-b_HST_STIS_G430L_52X2_Sing+2016_transmission_spectrum_1.txt"),
    # ]


    settings = {

    }


    target = "WASP-39 b"

    print(target)

    p = write_par_file(path_list, target=target, fastchem=False, ace=True, synthetic=True)

    print(p)


