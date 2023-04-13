import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, Callable
from tqdm import tqdm

import taurex.log
import logging
# taurex.log.disableLogging()
from taurex.parameter import ParameterParser
from taurex import OutputSize
from taurex.output.hdf5 import HDF5Output
from taurex.util.output import store_contributions

import mpi4py
mpi4py.rc.initialize = True  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically

from mpi4py import MPI # import the 'MPI' module


WDIR = Path().cwd().parent

def run_retrieval(input_file_path=None, output_file_path=None):
    if input_file_path is None:
        input_file_path = str(WDIR / "data/retrievals/DEFAULT/default_ultranest.par")
    if output_file_path is None:
        output_file_path = str(WDIR / "data/retrievals/DEFAULT/default_out.hdf5")

    # setup config parser
    pp = ParameterParser()
    pp.read(input_file_path)
    pp.setup_globals()

    # setup observations
    observation = pp.generate_observation()
    binning = pp.generate_binning()

    # make fw model
    model = pp.generate_appropriate_model(obs=observation)
    model.build()

    wngrid = None

    if binning == 'observed' and observation is None:
        logging.critical('Binning selected from Observation yet None provided')
        quit()

    if binning is None:
        if observation is None or observation == 'self':
            binning = model.defaultBinner()
            wngrid = model.nativeWavenumberGrid
        else:
            binning = observation.create_binner()
            wngrid = observation.wavenumberGrid
    else:
        if binning == 'native':
            binning = model.defaultBinner()
            wngrid = model.nativeWavenumberGrid
        elif binning == 'observed':
            binning = observation.create_binner()
            wngrid = observation.wavenumberGrid
        else:
            binning, wngrid = binning

    instrument = pp.generate_instrument(binner=binning)

    num_obs = 1
    if instrument is not None:
        instrument, num_obs = instrument

    if observation == 'self' and instrument is None:
        logging.getLogger('taurex').critical(
            'Instrument nust be specified when using self option')
        raise ValueError('No instruemnt specified for self option')

    inst_result = None
    if instrument is not None:
        inst_result = instrument.model_noise(
            model, model_res=model.model(), num_observations=num_obs)

    # Observation on self
    if observation == 'self':
        from taurex.data.spectrum import ArraySpectrum
        from taurex.util.util import wnwidth_to_wlwidth
        inst_wngrid, inst_spectrum, inst_noise, inst_width = inst_result

        inst_wlgrid = 10000 / inst_wngrid

        inst_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)
        observation = ArraySpectrum(
            np.vstack([inst_wlgrid, inst_spectrum,
                       inst_noise, inst_wlwidth]).T)
        binning = observation.create_binner()

    instrument = pp.generate_instrument(binner=binning)

    num_obs = 1
    if instrument is not None:
        instrument, num_obs = instrument

    if observation == 'self' and instrument is None:
        logging.getLogger('taurex').critical(
            'Instrument nust be specified when using self option')
        raise ValueError('No instruemnt specified for self option')

    inst_result = None
    if instrument is not None:
        inst_result = instrument.model_noise(
            model, model_res=model.model(), num_observations=num_obs)

    # Observation on self
    if observation == 'self':
        from taurex.data.spectrum import ArraySpectrum
        from taurex.util.util import wnwidth_to_wlwidth
        inst_wngrid, inst_spectrum, inst_noise, inst_width = inst_result

        inst_wlgrid = 10000 / inst_wngrid

        inst_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)
        observation = ArraySpectrum(
            np.vstack([inst_wlgrid, inst_spectrum,
                       inst_noise, inst_wlwidth]).T)
        binning = observation.create_binner()

    # setup optimizer, not the changes
    optimizer = pp.generate_optimizer()  # multi_nest_path=multi_nest_path)
    optimizer.set_model(model)
    optimizer.set_observed(observation)
    pp.setup_optimizer(optimizer)

    # output hdf5
    with HDF5Output(output_file_path) as o:
        model.write(o)

    # solve problem
    output_size = OutputSize.heavy

    # MPI.Init()  # manual initialization of the MPI environment
    solution = optimizer.fit(output_size=output_size)
    MPI.Finalize()  # manual finalization of the MPI environment

    # apply solution to fw model parameters
    for _, optimized, _, _ in optimizer.get_solution():
        optimizer.update_model(optimized)
        break
    result = model.model()

    # write output
    with HDF5Output(output_file_path, append=True) as o:
        out = o.create_group('Output')
        if observation is not None:
            obs = o.create_group('Observed')
            observation.write(obs)

        profiles = model.generate_profiles()
        spectrum = binning.generate_spectrum_output(result,
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


if __name__ == "__main__":
    run_retrieval()
    print("============================ DONE ============================")
