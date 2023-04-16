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
import seaborn as sns
from corner import corner

plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams['figure.dpi'] = 300


import taurex.log
# taurex.log.disableLogging()


from taurex.parameter import ParameterParser
from taurex import OutputSize
from taurex.output.hdf5 import HDF5Output
from taurex.util.output import store_contributions
from taurex_offset import OffsetSpectra
import inspect


WDIR = Path().cwd().parent

PLANET_DB_PATH = str(WDIR / "data/planet_database_composite.csv")
OPACITY_PATH = str(WDIR / "data/Input/xsec/xsec_sampled_R15000_0.3-15")
CIA_PATH = str(WDIR / "data/Input/cia/hitran")
KTABLE_PATH = str(WDIR / "data/Input/ktables/R100")
MOLECULE_PATH = str(WDIR / "data/molecule_db.json")

SPECTRA_BE_PATH = str(WDIR / "data/SpectraBE")
SPECTRA_LW_PATH = str(WDIR / "data/taurex_lightcurves_LW")

def show_plugins():
    from taurex.parameter.classfactory import ClassFactory
    from taurex.log import setLogLevel
    import logging
    setLogLevel(logging.ERROR)

    successful_plugins, failed_plugins = ClassFactory().discover_plugins()

    print('\nSuccessfully loaded plugins')
    print('---------------------------')
    for k, v in successful_plugins.items():
        print(k)

    print('\n\nFailed plugins')
    print('---------------------------')
    for k, v in failed_plugins.items():
        print(k)
        print(f'Reason: {v}')

    print('\n')

if __name__ == "__main__":
    path = str(WDIR / "data/retrievals/DEFAULT/default.par")
    output_file = str(WDIR / "data/retrievals/DEFAULT/default_out.hdf5")

    pp = ParameterParser()
    pp.read(path)

    pp.setup_globals()

    observation = pp.generate_observation()
    binning = pp.generate_binning()

    # TODO:
    """
    This is currently not well implemented as "observation" is not properly passed to the factory.create_model() method.
    This causes a a misrecognition of the kwargs that need to be passed to:
        TransmissionModel(SimpleForwardModel(ForwardModel))
    Where both TransmissionModel and SimpleForwardModel do not have observation as a properly defined kwarg
    
    This is a bit of a mess so I might have missed changes and issues, especially in optimizer.py or parameterparser.py
    
    TODO: 
    Stopgap changes / bad solution only for taurex_offset:
    
    factory.create_model():
        ...
        log.debug('Chosen_model is {}'.format(klass))
        kwargs = get_keywordarg_dict(klass, is_mixin)
        try:  # TODO: needs to be added
            from taurex_offset import OffsetSpectra
            if isinstance(observation, OffsetSpectra):
                kwargs["observation"] = observation
        except ImportError:
            pass
    
        log.debug('Model kwargs {}'.format(kwargs))
        log.debug('---------------{} {}--------------'.format(gas,
                                                              gas.activeGases))
        ...
    
    model.SimpleForwardModel:
        __init__():
            def __init__(self, name,
                         planet=None,
                         star=None,
                         pressure_profile=None,
                         temperature_profile=None,
                         chemistry=None,
                         observation=None,
                         nlayers=100,
                         atm_min_pressure=1e-4,
                         atm_max_pressure=1e6):
                super().__init__(name)
        
                self._planet = planet
                self._star = star
                self._pressure_profile = pressure_profile
                self._temperature_profile = temperature_profile
                self._chemistry = chemistry
                self._observation = observation
                self.debug('Passed: %s %s %s %s %s', planet, star, pressure_profile,
                           temperature_profile, chemistry)
            ...
        
        collect_fitting_parameters():
            ...
            for contrib in self.contribution_list:
                self._fitting_parameters.update(contrib.fitting_parameters())
    
            self._fitting_parameters.update(self._observation.fitting_parameters())
    
            self.debug('Available Fitting params: %s',
                       list(self._fitting_parameters.keys()))
            ...
        
    model.TransmissionModel:
        __init__():
            def __init__(self,
                     planet=None,
                     star=None,
                     pressure_profile=None,
                     temperature_profile=None,
                     chemistry=None,
                     observation=None, # TODO: needs to be added
                     nlayers=100,
                     atm_min_pressure=1e-4,
                     atm_max_pressure=1e6,
                     new_path_method=False):
    
                super().__init__(self.__class__.__name__, planet,
                                 star,
                                 pressure_profile,
                                 temperature_profile,
                                 chemistry,
                                 observation,  # TODO: needs to be added
                                 nlayers,
                                 atm_min_pressure,
                                 atm_max_pressure)
                self.new_method = new_path_method
        
        
    """

    model = pp.generate_appropriate_model(obs=observation)
    model.build()

    instrument = pp.generate_instrument(binner=binning)

    optimizer = pp.generate_optimizer()

    actual_fitting_para = pp.generate_fitting_parameters()

    optimizer.set_model(model)
    optimizer.set_observed(observation)
    pp.setup_optimizer(optimizer)

    output_size = OutputSize.heavy

    with HDF5Output(output_file) as o:
        model.write(o)

    solution = optimizer.fit(output_size=output_size)

    for _, optimized, _, _ in optimizer.get_solution():
        optimizer.update_model(optimized)
        break

    result = model.model()

    with HDF5Output(output_file, append=True) as o:

        out = o.create_group('Output')
        if observation is not None:
            obs = o.create_group('Observed')
            observation.write(obs)

        profiles = model.generate_profiles()
        spectrum = \
            binning.generate_spectrum_output(result,
                                             output_size=output_size)

        try:
            spectrum['Contributions'] = \
                store_contributions(binning, model,
                                    output_size=output_size - 3)
        except Exception:
            pass

        if solution is not None:
            out.store_dictionary(solution, group_name='Solutions')
            priors = {}
            priors['Profiles'] = profiles
            priors['Spectra'] = spectrum
            out.store_dictionary(priors, group_name='Priors')
        else:
            out.store_dictionary(profiles, group_name='Profiles')
            out.store_dictionary(spectrum, group_name='Spectra')

        if optimizer:
            optimizer.write(o)