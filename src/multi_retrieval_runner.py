from pathlib import Path
from src.par_file_writer import write_par_file, get_target_data
from src.retrieval_runner import run_retrieval
WDIR = Path().cwd().parent

import taurex.log
import logging
taurex.log.disableLogging()
import traceback

import itertools

import sys
from io import StringIO

import resource
import platform
import sys

from mpi4py import MPI

def setup_joint_retrieval_paths(input_dir=None, input_list=None, which=None):
    if input_dir is None and input_list is None:
        raise NotImplementedError
    if input_dir is not None:
        spectra = Path(input_dir).glob("*transmission_spectrum*")
        spectra = list(spectra)
    elif input_list is not None:
        spectra = [Path(p) for p in input_list]
    else:
        raise NotImplementedError

    out_file_list = []
    if which is None:
        for i in range(1, len(spectra) + 1):
            group = list(itertools.combinations(spectra, r=i))

            try:
                if isinstance(group[0], (list, tuple)):
                    group = [list(e) for e in group]
            except KeyError:
                pass

            out_file_list = out_file_list + group
    else:
        raise NotImplementedError

    return out_file_list

def main(file_dir=None, input_list=None, target_name=None, fastchem=False, ace=False, synthetic=True):
    files = setup_joint_retrieval_paths(input_dir=file_dir, input_list=input_list)

    target = get_target_data(target_name)

    par_file_paths = []

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for f in files:
        outf = write_par_file(f, target=target, fastchem=fastchem, ace=ace, synthetic=synthetic)
        par_file_paths.append(outf)

    comm.Barrier()
    par_file_paths = comm.bcast(par_file_paths, root=0)
    comm.Barrier()

    for path in par_file_paths:
        run_retrieval(input_file_path=str(path))
        comm.Barrier()

if __name__ == "__main__":
    # test_dir = WDIR / "data/synthetic_spectra/WASP-121b"
    # target_name = "WASP-39 b"
    synthetic = False
    ace = True
    fastchem = False
    # main(test_dir, target_name, synthetic=synthetic, fastchem=fastchem, ace=ace)

    names = [
        # "WASP-19 b",
        # "WASP-17 b",
        # "WASP-12 b",
        # "HD 189733 b",
        # "HAT-P-26 b",
        # "HAT-P-12 b",
        # "HAT-P-1 b",
    ]

    _dirs = [
        # "WASP-19b",
        # "WASP-17b",
        # "WASP-12b",
        # "HD189733b",
        # "HAT-P-26b",
        # "HAT-P-12b",
        # "HAT-P-1b",
    ]

    # dirs = [WDIR / "data/synthetic_spectra" / d for d in _dirs]

    names = [
        "WASP-39 b",
        "WASP-121 b",
    ]

    _dirs = [
        "WASP-39b",
        "WASP-121b",
    ]

    files = [
        [
            'WASP-121-b_HST_WFC3_G141_GRISM256_Evans+2016.txt',
            "WASP-121-b_HST_STIS_G430L_52X2_Sing+2019.txt"
        ],
        [
            "WASP-39-b_HST_WFC3_G141_GRISM256_Wakeford+2018.txt",
            "WASP-39-b_HST_STIS_G430L_52X2_Sing+2016.txt"
        ],
    ]

    dirs = [[WDIR / "data/taurex_lightcurves_LW" / f for f in d] for d in files]

    for direct, name, file_list in zip(dirs, names, files):
        try:
            main(file_dir=None, target_name=name, input_list=file_list,
                 synthetic=synthetic, fastchem=fastchem, ace=ace)
        except BaseException as e:
            print(f"\n\n\n\n\n"
                  f"==========================================================================================\n"
                  f"Could not complete run for {name} from {direct}.\n"
                  f"{e}\n"
                  f"=========================================================================================="
                  f"\n\n\n\n\n")
            pass
