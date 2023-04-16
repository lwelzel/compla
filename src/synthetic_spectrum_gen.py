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

WDIR = Path().cwd().parent

if __name__ == "__main__":
    input_file_path = str(WDIR / "data/synthetic_spectra/DEFAULT/default_synthetic_ultranest1.par")
    output_file_path = str(WDIR / "data/synthetic_spectra/DEFAULT/default_synthetic_out1.hdf5")
    output_spectrum_file_path = str(WDIR / "data/synthetic_spectra/DEFAULT/default_synthetic_spectrum1.txt")
    output_size = OutputSize.heavy

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
            model,
            model_res=model.model(),
            num_observations=num_obs
        )

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

        model = model.model(wngrid=inst_wlwidth)

    # output hdf5
    with HDF5Output(output_file_path) as o:
        model.write(o)

    result = model.model()

    from taurex.util.util import wnwidth_to_wlwidth, compute_bin_edges

    save_wnwidth = compute_bin_edges(wngrid)[1]
    save_wl = 10000 / wngrid
    save_wlwidth = wnwidth_to_wlwidth(wngrid, save_wnwidth)
    save_model = binning.bin_model(result)[1]
    save_error = np.zeros_like(save_wl)
    if inst_result is not None:
        inst_wngrid, inst_spectrum, inst_noise, inst_width = inst_result

        save_model = inst_spectrum
        save_wl = 10000 / inst_wngrid

        save_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)

        save_error = inst_noise

    np.savetxt(output_spectrum_file_path,
               np.vstack((save_wl, save_model, save_error,
                          save_wlwidth)).T)


    # Output taurex data
    with HDF5Output(output_file_path, append=True) as o:

        out = o.create_group('Output')
        if observation is not None:
            obs = o.create_group('Observed')
            observation.write(obs)

        profiles = model.generate_profiles()
        spectrum = \
            binning.generate_spectrum_output(result,
                                             output_size=output_size)

        if inst_result is not None:
            spectrum['instrument_wngrid'] = inst_result[0]
            spectrum['instrument_wnwidth'] = inst_result[-1]
            spectrum['instrument_wlgrid'] = 10000 / inst_result[0]
            spectrum['instrument_spectrum'] = inst_result[1]
            spectrum['instrument_noise'] = inst_result[2]

        try:
            spectrum['Contributions'] = \
                store_contributions(binning, model,
                                    output_size=output_size - 3)
        except Exception:
            pass
