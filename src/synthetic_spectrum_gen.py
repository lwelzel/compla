import os
import pathlib

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
from par_file_writer import write_par_file
from par_file_writer import get_target_data

WDIR = Path().cwd().parent


def make_synthetic_spectrum_from(input_file_path=None,
                                 output_file_path=None,
                                 output_spectrum_file_path=None,
                                 num_obs=1,
                                 offset=0.,
                                 uid=0,
                                 force_input_error=False, input_error=None):

    if input_file_path is None:
        input_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014_time-2023-04-19-09-19-39.par")
    if output_file_path is None:
        output_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014_time-2023-04-19-09-19-39.hdf5")
    if output_spectrum_file_path is None:
        output_spectrum_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014/synthetic_HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014_time-2023-04-19-09-19-39_transmission_spectrum_0.txt")

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

    if instrument is not None:
        instrument, num_obs = instrument

    if True and instrument is None:  # if observation == 'self' and instrument is None:
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

    if inst_result is not None:
        inst_wngrid, inst_spectrum, inst_noise, inst_width = inst_result

        save_model = inst_spectrum
        save_wl = 10000 / inst_wngrid

        save_wlwidth = wnwidth_to_wlwidth(inst_wngrid, inst_width)

        save_error = inst_noise

    if force_input_error and input_error is not None:
        save_error = np.flip(input_error)

    np.savetxt(output_spectrum_file_path,
               np.vstack((save_wl,
                          save_model + np.mean(save_model) * offset,
                          save_error,
                          save_wlwidth)).T)

    # Output taurex data
    with HDF5Output(output_file_path, append=True) as o:

        out = o.create_group('Output')
        if observation is not None:
            obs = o.create_group('Observed')
            observation.write(obs)

def make_synthetic_spectrum(name, base_spectrum_list, offset=0., fastchem=True):
    target = get_target_data(name)

    path = str(Path(WDIR / f"data/synthetic_spectra/{name.replace(' ', '')}"))
    os.makedirs(path, exist_ok=True)

    if isinstance(base_spectrum_list, list) and len(base_spectrum_list) > 1 and isinstance(offset, float):
        offset = np.array([offset for __ in base_spectrum_list])
        offset[0] = 0.
    if isinstance(offset, float):
        offset = [offset]
    if isinstance(base_spectrum_list, str):
        base_spectrum_list = [base_spectrum_list]

    for i, (spectrum, single_offset) in enumerate(zip(base_spectrum_list, offset)):
        in_name = Path(spectrum).stem

        filename = f"synthetic_{in_name}_{i}.par"

        par_file_path = str(Path(path) / filename)

        write_par_file(spectrum,
                       target=target, fastchem=fastchem, synthetic=True,
                       path=path, filename=filename,
                       comments=["Synthetic spectrum with forced errors like in the input file."])

        base_spectrum = np.loadtxt(spectrum).T
        base_spectrum_errors = base_spectrum[2].flatten()

        spec_file_name = f"synthetic_{in_name}_transmission_spectrum_{i}.txt"

        h5_file_name = f"synthetic_{in_name}_data_{i}.hdf5"

        make_synthetic_spectrum_from(input_file_path=par_file_path,
                                     output_file_path=str(Path(path) / h5_file_name),
                                     output_spectrum_file_path=str(Path(path) / spec_file_name),
                                     offset=single_offset,
                                     force_input_error=True, input_error=base_spectrum_errors,
                                     )

if __name__ == "__main__":
    # num_obs = 1
    # input_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/"
    #                              "HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016/HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016_time-2023-04-19-09-19-47.par")
    # # input_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/"
    # #                              "HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016/HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016_time-2023-04-19-09-19-47.par")
    #
    # output_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014")
    # output_spectrum_file_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Nikolov+2014/")

    # path_list = [
    #     str(WDIR / "data/taurex_lightcurves_LW" / "WASP-121-b_HST_WFC3_G141_GRISM256_Evans+2016.txt"),
    #     str(WDIR / "data/taurex_lightcurves_LW" / "WASP-121-b_HST_STIS_G430L_52X2_Sing+2019.txt"),
    # ]
    #
    path_list = [
        str(WDIR / "data/taurex_lightcurves_LW" / "WASP-39-b_HST_WFC3_G141_GRISM256_Wakeford+2018.txt"),
        str(WDIR / "data/taurex_lightcurves_LW" / "WASP-39-b_HST_STIS_G430L_52X2_Sing+2016.txt"),
        str(WDIR / "data/taurex_lightcurves_LW" / "WASP-39-b_HST_STIS_G430L_52X2_Fischer+2016_NO-BW.txt"),
    ]

    make_synthetic_spectrum("WASP-39 b", base_spectrum_list=path_list, fastchem=True)
