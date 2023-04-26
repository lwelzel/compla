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

from mpi4py import MPI  # import the 'MPI' module

WDIR = Path().cwd().parent

def run_retrieval(input_file_path=None, output_file_path=None, resume=False):
    if input_file_path is None:
        raise NotImplementedError
    if output_file_path is None:
        output_file_path = str(Path(input_file_path).parent / Path(input_file_path).stem / Path(input_file_path).name).replace(".par", "_output.hdf5")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()

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
    if resume:
        optimizer.resume = True
    optimizer.set_model(model)
    optimizer.set_observed(observation)
    pp.setup_optimizer(optimizer)
    if resume:
        optimizer.resume = True

    # output hdf5
    #  TODO:
    if not resume:
        output_file_path = str(Path(output_file_path).parent.parent / (Path(output_file_path).stem + ".hdf5"))
        with HDF5Output(output_file_path) as o:
            model.write(o)

    # solve problem
    output_size = OutputSize.light

    comm.Barrier()
    solution = optimizer.fit(output_size=output_size)
    comm.Barrier()

    # apply solution to fw model parameters
    for _, optimized, _, _ in optimizer.get_solution():
        optimizer.update_model(optimized)
        break
    result = model.model()

    comm.Barrier()

    # write output
    with HDF5Output(output_file_path, append=True) as o:
        out = o.create_group('Output')
        try:
            if observation is not None:
                obs = o.create_group('Observed')
                observation.write(obs)
        except Exception:
            pass
        try:
            profiles = model.generate_profiles()
            spectrum = \
                binning.generate_spectrum_output(result,
                                                 output_size=output_size)
        except Exception:
            pass
        try:
            if inst_result is not None:
                spectrum['instrument_wngrid'] = inst_result[0]
                spectrum['instrument_wnwidth'] = inst_result[-1]
                spectrum['instrument_wlgrid'] = 10000 / inst_result[0]
                spectrum['instrument_spectrum'] = inst_result[1]
                spectrum['instrument_noise'] = inst_result[2]
        except Exception:
            pass
        try:
            spectrum['Contributions'] = \
                store_contributions(binning, model,
                                    output_size=output_size - 3)
        except Exception:
            pass

        if solution is not None:
            try:
                out.store_dictionary(solution, group_name='Solutions')
            except Exception:
                pass
            priors = {}
            try:
                priors['Profiles'] = profiles
            except Exception:
                pass
            try:
                priors['Spectra'] = spectrum
            except Exception:
                pass
            try:
                out.store_dictionary(priors, group_name='Priors')
            except Exception:
                pass
        else:
            out.store_dictionary(profiles, group_name='Profiles')
            out.store_dictionary(spectrum, group_name='Spectra')

        try:
            optimizer.write(o)
        except Exception:
            pass

    comm.Barrier()

if __name__ == "__main__":

    test_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016,TM1/HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016,TM1_time-2023-04-18-10-07-50.par")

    run_retrieval(input_file_path=test_path)
