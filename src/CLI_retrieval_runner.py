import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, Callable
from tqdm import tqdm
import os
import subprocess

import taurex.log
import logging
# taurex.log.disableLogging()
from taurex.parameter import ParameterParser
from taurex import OutputSize
from taurex.output.hdf5 import HDF5Output
from taurex.util.output import store_contributions

import os
import re
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from configobj import ConfigObj

from mpi4py import MPI  # import the 'MPI' module

WDIR = Path().cwd().parent

def run_CLI_command(command):
    out = os.system(command)
    return out

def compare_par_files(file1, file2, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = ["log_dir"]
    # Read and parse both .par files
    config1 = ConfigObj(file1)
    config2 = ConfigObj(file2)

    # Compare the ConfigObj objects and return True if they are the same, or False otherwise
    return {k:v for k,v in config1.items() if k not in ignore_keys} == {k:v for k,v in config2.items() if k not in ignore_keys}

def find_closest_file(input_file_path, extension=None, use_newest=False):
    input_file = Path(input_file_path)
    parent_directory = input_file.parent

    if not extension:
        extension = input_file.suffix

    matching_files = [str(file) for file in parent_directory.rglob(f'*{extension}')]

    def similarity(a, b):
        return SequenceMatcher(None, Path(a).stem, Path(b).stem).ratio()

    sorted_files = sorted(matching_files, key=lambda x: similarity(input_file.stem, Path(x).stem), reverse=True)

    for f in sorted_files:
        print(f)

    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}')
    input_stem_no_datetime = re.sub(date_pattern, '', input_file.stem)

    if max([similarity(input_stem_no_datetime, re.sub(date_pattern, '', Path(p).stem)) for p in matching_files]) == 1:
        files_with_date = [file for file in sorted_files if date_pattern.search(file)]

        if files_with_date:
            file_dates = [datetime.strptime(date_pattern.search(file).group(), "%Y-%m-%d-%H-%M-%S") for file in files_with_date]

            if use_newest:
                file_index = file_dates.index(max(file_dates))
            else:
                input_datetime = datetime.strptime(date_pattern.search(input_file.name).group(), "%Y-%m-%d-%H-%M-%S")
                time_differences = [abs(input_datetime - file_date) for file_date in file_dates]
                file_index = time_differences.index(min(time_differences))

            return files_with_date[file_index], similarity(Path(input_file).stem, Path(files_with_date[file_index]).stem)

    return sorted_files[0], similarity(Path(input_file).stem, Path(sorted_files[0]).stem)

def find_resume_file(query_file, query_extension=None, use_newest=False, tolerance=0.9):
    proposed_file, score = find_closest_file(query_file, extension=query_extension, use_newest=use_newest)
    if score < tolerance:
        print(f"Tolerance in file name was violated: {score:.2f} < {tolerance:.2f}. Returning input path.")
        return query_file

    # test if file contents are the same
    if Path(query_file).suffix == Path(proposed_file).suffix and compare_par_files(query_file, proposed_file):
        return proposed_file
    else:
        return proposed_file

def run_retrieval_CLI(input_file_path=None, output_file_path=None, resume=False):
    if input_file_path is None:
        raise NotImplementedError
    if output_file_path is None:
        output_file_path = str(Path(input_file_path).parent / Path(input_file_path).stem / Path(input_file_path).name).replace(".par", "_output.hdf5")

    cli_env_activation = "conda activate compla"
    cli_cd = f'cd "{str(WDIR / "src")}"'
    cli_command_standard = 'mpirun -n 8 python taurex --input "{0}" --output_file "{1}" --retrieval --light'
    cli_command_no_output = 'mpirun -n 8 python taurex --input "{0}" --retrieval --light'
    cli_command_no_output = [f'mpirun', '-n 8', 'python taurex', '--input "{str(input_file_path)}"', '--retrieval', '--light']
    cli_output_only = 'taurex --input "{0}" --output_file "{1}" --retrieval --light'  # TODO: resume=True

    __ = subprocess.run([cli_env_activation])
    __ = subprocess.run([cli_cd])
    __ = subprocess.run(cli_command_no_output)

def solve_retrieval(input_file_path=None, output_file_path=None, resume=False):
    if input_file_path is None:
        raise NotImplementedError
    if output_file_path is None:
        output_file_path = str(Path(input_file_path).parent / Path(input_file_path).stem / Path(input_file_path).name).replace(".par", "_output.hdf5")

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
    output_file_path = str(Path(output_file_path).parent.parent / (Path(output_file_path).stem + ".hdf5"))
    with HDF5Output(output_file_path) as o:
        model.write(o)

    # solve problem
    output_size = OutputSize.light

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    solution = optimizer.fit(output_size=output_size)
    comm.Barrier()

    return output_file_path


def write_retrieval_result(input_file_path, output_file_path, resume=True):
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

    optimizer.resume = True
    optimizer.set_model(model)
    optimizer.set_observed(observation)
    pp.setup_optimizer(optimizer)

    optimizer.resume = True

    # output hdf5
    output_file_path = str(Path(output_file_path).parent.parent / (Path(output_file_path).stem + ".hdf5"))

    # solve problem
    output_size = OutputSize.light

    solution = optimizer.fit(output_size=output_size, resume=resume)

    # apply solution to fw model parameters
    for _, optimized, _, _ in optimizer.get_solution():
        optimizer.update_model(optimized)
        break
    result = model.model()

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

if __name__ == "__main__":

    test_path = str(WDIR / "data/synthetic_spectra/HAT-P-1b/HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016,TM1/HAT-P-1b_HST_STIS_G430L_52X2_Sing+2016,TM1_time-2023-04-19-14-44-45.par")

    # run_retrieval(input_file_path=test_path)

    closest_match = find_resume_file(test_path, query_extension=".bogus_hdf5")
    print()
    print(test_path)
    print(closest_match)


