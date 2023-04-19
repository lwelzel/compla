from pathlib import Path
from src.par_file_writer import write_par_file, get_target_data
from src.retrieval_runner import run_retrieval
WDIR = Path().cwd().parent

if __name__ == "__main__":
    test1_files = [
        str(WDIR / "data/synthetic_spectra/HAT-P-1b" / "synthetic_HAT-P-1-b_HST_STIS_G430L_52X2_Nikolov+2014_transmission_spectrum_0.txt"),
    ]

    test2_files = [
        str(WDIR / "data/synthetic_spectra/HAT-P-1b" / "synthetic_HAT-P-1-b_HST_STIS_G430L_52X2_Sing+2016_transmission_spectrum_1.txt"),
    ]

    test3_files = [
        str(WDIR / "data/synthetic_spectra/HAT-P-1b" / "synthetic_HAT-P-1-b_HST_STIS_G430L_52X2_Nikolov+2014_transmission_spectrum_0.txt"),
        str(WDIR / "data/synthetic_spectra/HAT-P-1b" / "synthetic_HAT-P-1-b_HST_STIS_G430L_52X2_Sing+2016_transmission_spectrum_1.txt"),
    ]

    files = [
        test1_files,
        test2_files,
        test3_files
    ]

    target = get_target_data("HAT-P-1 b")

    # TODO: generate spectra?

    par_file_paths = []

    # write par files
    for f in files:
        outf = write_par_file(f, target=target, fastchem=True, synthetic=True)
        par_file_paths.append(outf)

    for path in par_file_paths:
        run_retrieval(input_file_path=str(path))