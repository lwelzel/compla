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