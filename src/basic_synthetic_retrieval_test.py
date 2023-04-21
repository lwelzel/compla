from pathlib import Path
from src.par_file_writer import write_par_file, get_target_data
from src.retrieval_runner import run_retrieval
WDIR = Path().cwd().parent

# import taurex.log
# import logging
# logging.basicConfig(level=logging.DEBUG)
import traceback

import sys
from io import StringIO

import resource
import platform
import sys

class CaptureStdout:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self.captured_stdout = StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout

    def get_captured_stdout(self):
        return self.captured_stdout.getvalue()

def find_error_line(error_message, script_path):
    with open(script_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if error_message in line:
            return i + 1

    return None

def memory_limit(percentage: float):
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))
    
def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory(percentage=0.75):
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator

# @memory(percentage=0.95)
def main():
    test1_files = [
        str(WDIR / "data/synthetic_spectra/WASP-121b" / "synthetic_WASP-121-b_HST_WFC3_G141_GRISM256_Evans+2016_transmission_spectrum_0.txt"),
    ]

    test2_files = [
        str(WDIR / "data/synthetic_spectra/WASP-121b" / "synthetic_WASP-121-b_HST_STIS_G430L_52X2_Sing+2019_transmission_spectrum_1.txt"),
    ]

    test3_files = [
        str(WDIR / "data/synthetic_spectra/WASP-121b" / "synthetic_WASP-121-b_HST_WFC3_G141_GRISM256_Evans+2016_transmission_spectrum_0.txt"),
        str(WDIR / "data/synthetic_spectra/WASP-121b" / "synthetic_WASP-121-b_HST_STIS_G430L_52X2_Sing+2019_transmission_spectrum_1.txt"),
    ]

    files = [
        test1_files,
        test2_files,
        test3_files
    ]

    target = get_target_data("HAT-P-1 b")

    # TODO: generate spectra?

    par_file_paths = []

    # with CaptureStdout() as capture:
    #     # write par files
    for f in files:
        outf = write_par_file(f, target=target, fastchem=True, synthetic=True)
        par_file_paths.append(outf)

    for path in par_file_paths:
        run_retrieval(input_file_path=str(path))

    # captured_output = capture.get_captured_stdout()
    #
    # error_message = "stat: path should be string, bytes, os.PathLike or integer, not NoneType"
    #
    # if error_message in captured_output:
    #     script_path = "your_script_path.py"
    #     error_line = find_error_line(error_message, script_path)
    #
    #     if error_line is not None:
    #         print(f"Found error message in script '{script_path}' at line {error_line}")
    #     else:
    #         print("Error message not found in the script.")
    # else:
    #     print("No error message detected.")

if __name__ == "__main__":
    main()
