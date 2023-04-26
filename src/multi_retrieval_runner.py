from pathlib import Path
from src.par_file_writer import write_par_file, get_target_data
from src.retrieval_runner import run_retrieval
WDIR = Path().cwd().parent

import taurex.log
import logging
# taurex.log.disableLogging()
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

def main(file_dir, target_name, fastchem=False, ace=False, synthetic=True):
    files = setup_joint_retrieval_paths(input_dir=file_dir)

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
    synthetic = True
    ace = True
    fastchem = False
    # main(test_dir, target_name, synthetic=synthetic, fastchem=fastchem, ace=ace)

    names = [
        "WASP-19 b",
        "WASP-17 b",
        "WASP-12 b",
        "HD-189733 b",
        "HAT-P-26 b",
        "HAT-P-12 b",
        "HAT-P-1 b",
    ]

    _dirs = [
        "WASP-19b",
        "WASP-17b",
        "WASP-12b",
        "HD-189733b",
        "HAT-P-26b",
        "HAT-P-12b",
        "HAT-P-1b",
    ]

    dirs = [WDIR / "data/synthetic_spectra" / d for d in _dirs]

    for direct, name in dirs, names:
        try:
            main(direct, name, synthetic=synthetic, fastchem=fastchem, ace=ace)
        except BaseException:
            pass
