import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from copy import deepcopy
from corner import corner
from pathlib import Path

from taurex.binning.fluxbinner import FluxBinner
from taurex.util.util import wnwidth_to_wlwidth

WDIR = Path().cwd().parent



def bin_spectrum(low_res, high_res):
    transmission_spectra = high_res.sort_values(by='wavelength')
    low_res = low_res.sort_values(by='wavelength')
    low_res = low_res.reset_index(drop=True)

    binner = FluxBinner(low_res['wavelength'].values)

    _, binned_transit_depth, _, _ = binner.bindown(transmission_spectra['wavelength'].values,
                                                   transmission_spectra['depth'].values)

    return np.array(binned_transit_depth)

def plot_spectrum_into_ax(hdf5file=None, outfile=None, ax=None):
    if hdf5file is None:
        pass
    if outfile is None:
        pass

    if ax is None:
        ax = plt.gca()

    res = h5py.File(hdf5file, 'r')
    res_spec = res['Output']['Solutions']['solution0']['Spectra']
    factor = 1e2

    hr = pd.DataFrame(res_spec['native_wlgrid'][()], columns=['wavelength'])
    hr['depth'] = res_spec['native_spectrum'][()]

    bd = 0.005
    lr = pd.DataFrame(np.arange(0.6, 5, bd), columns=['wavelength'])
    lr['depth'] = bin_spectrum(lr.copy(), hr)

    hr_plus_std = hr.copy()
    hr_plus_std['depth'] = res_spec['native_spectrum'][()] + res_spec['native_std'][()]

    hr_minus_std = hr.copy()
    hr_minus_std['depth'] = res_spec['native_spectrum'][()] - res_spec['native_std'][()]

    lr['depth_m_std'] = bin_spectrum(lr.copy(), hr_minus_std)
    lr['depth_p_std'] = bin_spectrum(lr.copy(), hr_plus_std)

    try:
        ax.errorbar(res['Observed']['wlgrid'][()], res['Observed']['spectrum'][()] * factor,
                     yerr=res['Observed']['errorbars'][()] * factor,
                     fmt='o', c='k', capsize=5)
    except KeyError:
        pass
    ax.plot(lr['wavelength'], lr['depth'] * factor, c='r')
    ax.fill_between(lr['wavelength'], lr['depth_m_std'] * factor, lr['depth_p_std'] * factor, color='r', alpha=0.3)
    ax.set_xlim(0.85, 1.7)
    ax.set_ylim(2.01, 2.17)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Transit Depth [%]')
    return ax

def get_evidence(hdf5file=None):
    if hdf5file is None:
        pass
    res = h5py.File(hdf5file, 'r')
    evidence = deepcopy(res['Output']['Solutions']['solution0']['Statistics']['local log-evidence'][()])
    res.close()
    return evidence

def divergence_models(model_files, burn_in=0.1):
    fig = plt.figure(num="divergence_corner", constrained_layout=True)

    if not isinstance(burn_in, int):
        burn_in = 100

    colors = ["red", "blue", "green"]
    for i, file in enumerate(model_files):
        res = h5py.File(file, 'r')

        fit_para = res['Optimizer']['fit_parameter_names'][()]
        final_para = res['Output']['Solutions']['solution0']['fit_params']

        traces = res['Output']['Solutions']['solution0']['tracedata'][()]

        if not isinstance(burn_in, int):
            burn_in = int(burn_in * len(traces))

        traces = traces.T

        corner(
            data=traces[:, burn_in:],
            labels=fit_para,
            truths=final_para,
            fig=fig, quiet=True,
        )

    filename = Path(model_files[0]).parent / "divergence_corner.png"
    plt.savefig(filename, dpi=300)

def spectrum_differences(model_files, outfile=None):
    fig, ax = plt.subplots(1, 1, num="spectrum_differences", constrained_layout=True)

    for i, file in enumerate(model_files):
        ax = plot_spectrum_into_ax(file)

    filename = Path(model_files[0]).parent / "spectrum_differences.png"
    plt.savefig(filename, dpi=300)

