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

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams['figure.dpi'] = 300

import taurex.log
# taurex.log.disableLogging()
from taurex.cache import OpacityCache, CIACache
from taurex.data.profiles.temperature import Isothermal, Guillot2010, NPoint
from taurex.data import Planet
from taurex.data.stellar import PhoenixStar
from taurex.data.stellar import BlackbodyStar
from taurex.data.profiles.chemistry import TaurexChemistry
from taurex.data.profiles.chemistry import ConstantGas
from taurex.data.profiles.chemistry import TwoLayerGas
from taurex.model import TransmissionModel, EmissionModel
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution, HydrogenIon
from taurex.data.spectrum.observed import ObservedSpectrum
from astropy.io.fits.verify import VerifyWarning
from taurex.binning.fluxbinner import FluxBinner
from taurex.util.util import wnwidth_to_wlwidth
from taurex.optimizer.nestle import NestleOptimizer


WDIR = Path().cwd().parent

PLANET_DB_PATH = str(WDIR / "data/planet_database_composite.csv")
OPACITY_PATH = str(WDIR / "data/Input/xsec/xsec_sampled_R15000_0.3-15")
CIA_PATH = str(WDIR / "data/Input/cia/hitran")
MOLECULE_PATH = str(WDIR / "data/molecule_db.json")

SPECTRA_BE_PATH = str(WDIR / "data/SpectraBE")
SPECTRA_LW_PATH = str(WDIR / "data/taurex_lightcurves_LW")

OpacityCache().set_opacity_path(OPACITY_PATH)
CIACache().set_cia_path(CIA_PATH)

default_tm_settings = {

}

default_warning_filters = [
    {"category":DeprecationWarning, "module":''},
    {"category":FutureWarning, "module":''},
    {"category":RuntimeWarning, "module":''},
    {"category":Warning, "module":'numpy'},
]

warnings.simplefilter("always")
for f in default_warning_filters:
    warnings.filterwarnings("ignore", **f)

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


@dataclass
class AliasedDict(dict):
    aliases: dict

    def __init__(self, data: dict, aliases: dict):
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

def get_target_data(planet_name):
    exoplanet_database = pd.read_csv(PLANET_DB_PATH, comment="#", index_col=0)

    target_data = exoplanet_database.loc[exoplanet_database["pl_name"] == planet_name].to_dict(orient="records")[0]

    col_names = process_lines(PLANET_DB_PATH)
    col_names = {v: k for k, v in col_names.items()}

    target = AliasedDict(target_data, aliases=col_names)

    return target


def make_gasses(gas_input_list: Dict = None):
    if gas_input_list is None:
        gas_input_list = read_json_file(MOLECULE_PATH)
    else:
        gas_input_list = [*read_json_file(MOLECULE_PATH), *gas_input_list]

    h2_gas = TwoLayerGas(
        'H',
        mix_ratio_surface=1e-2,
        mix_ratio_top=0.5,
        mix_ratio_P=1e4
    )

    gasses = [h2_gas]

    types = {
        'ConstantGas': ConstantGas,
        'TwoLayerGas': TwoLayerGas,
        'HydrogenIon': HydrogenIon,
    }

    for i, gas in enumerate(gas_input_list):
        gasses.append(
            types[gas["type"]](
                gas["molecule"],
                mix_ratio=10 ** gas["abundance"]
            )
        )

    return gasses, gas_input_list


def build_tm_dynamic(target, settings: (Dict, Any) = None) -> TransmissionModel:
    isothermal = Isothermal(T=target["Equilibrium Temperature [K]"])

    planet = Planet(
        planet_radius=target["Planet Radius [Jupiter Radius]"],
        planet_mass=target["Planet Mass or Mass*sin(i) [Jupiter Mass]"],
    )

    star = BlackbodyStar(
        temperature=target["Stellar Effective Temperature [K]"],
        radius=target["Stellar Radius [Solar Radius]"],
        mass=target["Stellar Mass [Solar mass]"]
    )

    chemistry = TaurexChemistry(
        fill_gases=['H2', 'He'],
        ratio=0.17
    )

    gasses, gas_input_list = make_gasses()

    for gas in gasses:
        chemistry.addGas(gas)

    tm = TransmissionModel(
        planet=planet,
        temperature_profile=isothermal,
        chemistry=chemistry,
        star=star,
        atm_min_pressure=1e-4,
        atm_max_pressure=1e6,
        nlayers=30,
    )

    tm.add_contribution(AbsorptionContribution())

    tm.add_contribution(RayleighContribution())

    tm.add_contribution(CIAContribution(cia_pairs=['H2-H2', 'H2-He']))

    if "e-" in [gas['molecule'] for gas in gas_input_list]:
        tm.add_contribution(HydrogenIon())

    tm.build()

    return tm

if __name__ == "__main__":

    target = get_target_data("WASP-121 b")

    tm = build_tm_dynamic(target, settings={})

    for e in tm.fittingParameters:
        print(e)
    print(list(tm.fittingParameters.keys()))


    obs = ObservedSpectrum(SPECTRA_BE_PATH + "/WASP-121b_G141.txt")

    obin = obs.create_binner()

    plt.figure()
    plt.errorbar(obs.wavelengthGrid, obs.spectrum, obs.errorBar, label='Obs')
    plt.plot(obs.wavelengthGrid, obin.bin_model(tm.model(obs.wavenumberGrid))[1], label='TM')
    plt.legend()
    plt.show()

    opt = NestleOptimizer(num_live_points=50)


    opt.set_model(tm)


    opt.set_observed(obs)

    for e in tm.fittingParameters:
        opt.enable_fit(e)

    # opt.enable_fit('planet_radius')
    #
    # opt.enable_fit('T')
    #
    opt.set_boundary('T', [1000, 3500])

    opt.set_boundary('planet_radius', [0.8, 2.2])

    solution = opt.fit()
    # taurex.log.disableLogging()

    print(opt.fitting_parameters)

    print(opt.derived_parameters)

    for solution, optimized_map, optimized_value, values in opt.get_solution():
        opt.update_model(optimized_map)
        plt.figure()
        plt.errorbar(obs.wavelengthGrid, obs.spectrum, obs.errorBar, label='Obs')
        plt.plot(obs.wavelengthGrid, obin.bin_model(tm.model(obs.wavenumberGrid))[1], label='TM')
        plt.legend()
        plt.show()
