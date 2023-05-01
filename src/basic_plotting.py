import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
import re
import numpy.ma as ma

from copy import deepcopy
from corner import corner

from taurex.binning.fluxbinner import FluxBinner
from taurex.util.util import wnwidth_to_wlwidth

WDIR = Path().cwd().parent


def get_target_results(path=None):
    if path is None:
        path = Path(WDIR / "data/synthetic_spectra/WASP-39b")

    path = Path(path)

    paths = Path(path).rglob("*_output.hdf5")
    paths = list(paths)
    results = [""] * len(paths)

    for i, res in enumerate(paths):
        results[i] = h5py.File(res, "r")

    for i, res in enumerate(results):
        try:
            __ = res['Observed']
            __ = res['Output']['Solutions']
        except KeyError:
            print(f"{paths[i].name} does not have Observed or Solutions")

    return results, paths

def split_on_underscore(s):
    s = s.replace("spectral_element", "spectral-element")
    pattern = r'(?<!\()_(?!\))'
    temp_parts = re.split(pattern, s)
    parts = []
    i = 0
    while i < len(temp_parts):
        part = temp_parts[i]
        if '(' in part and ')' not in part:
            combined_parts = [part]
            i += 1
            while i < len(temp_parts) and ')' not in temp_parts[i]:
                combined_parts.append(temp_parts[i])
                i += 1
            combined_parts.append(temp_parts[i])
            parts.append('_'.join(combined_parts))
        else:
            parts.append(part)
        i += 1
    return parts


def bin_spectrum(low_res, high_res):
    transmission_spectra = high_res.sort_values(by='wavelength')
    low_res = low_res.sort_values(by='wavelength')
    low_res = low_res.reset_index(drop=True)

    binner = FluxBinner(low_res['wavelength'].values)

    _, binned_transit_depth, _, _ = binner.bindown(transmission_spectra['wavelength'].values,
                                                   transmission_spectra['depth'].values)

    return np.array(binned_transit_depth)

def plot_spectrum_into_ax(hdf5file=None, outfile=None, ax=None, label=None):
    if hdf5file is None:
        pass
    if outfile is None:
        pass

    if ax is None:
        ax = plt.gca()

    if isinstance(hdf5file, (str, Path)):
        res = h5py.File(hdf5file, 'r')
    elif isinstance(hdf5file, h5py.File):
        res = hdf5file
    else:
        raise NotImplementedError

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
                    fmt='D', c='k', capsize=5,
                    markersize=2.,
                    elinewidth=0.9,
                    )
    except KeyError:
        pass
    ax.plot(lr['wavelength'], lr['depth'] * factor, c='r', lw=0.8)
    ax.fill_between(lr['wavelength'], lr['depth_m_std'] * factor, lr['depth_p_std'] * factor, color='r', alpha=0.3)
    # ax.set_xlim(0.85, 1.7)
    # ax.set_ylim(2.01, 2.17)
    ax.set_ylabel('Transit Depth [%]')
    ax.set_title(label)
    return ax

def plot_spectra(path=None):
    # fig, ax = plt.subplots(1, 1, num="spectrum_differences", constrained_layout=True)
    #
    # for i, file in enumerate(results):
    #     plt.sca(ax)
    #     ax = plot_spectrum_into_ax(file)
    #
    # filename = Path(paths[0]).parent / "spectrum_differences.png"
    # plt.savefig(filename, dpi=300)

    results, paths = get_target_results(path=path)

    names = [p.stem for p in paths]
    split_names = [split_on_underscore(name) for name in names]

    labels = [
        f'{nlist[2].replace("instrument(", "").replace(")", "")}'.replace("_", "+")
        for nlist in split_names
    ]

    fig, axes = plt.subplots(len(results), 1,
                             sharex=True,
                             num="stacked_spectrum_differences", constrained_layout=True,
                             figsize=(len(results) * 4., len(results) * 4.))

    axes = np.array(axes).flatten()

    for i, (file, ax, label) in enumerate(zip(results, axes, labels)):
        plt.sca(ax)
        ax = plot_spectrum_into_ax(file, label=label)
    ax.set_xlabel(r'Wavelength [$\mu m$]')

    fig.suptitle(f"{str(Path(paths[0]).parent.parent.stem)}")

    filename = Path(paths[0]).parent.parent / "stacked_spectrum_differences.png"
    plt.savefig(filename, dpi=300)

def get_evidence(hdf5file=None):
    if hdf5file is None:
        pass
    if isinstance(hdf5file, (str, Path)):
        res = h5py.File(hdf5file, 'r')
    elif isinstance(hdf5file, h5py.File):
        res = hdf5file
    else:
        raise NotImplementedError

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

def merge_data_sets(data_sets):
    all_data = pd.DataFrame()
    for data in data_sets:
        df = pd.DataFrame(data)
        all_data = pd.concat([all_data, df], axis=0)
    return all_data

def make_struct_df(results, paths):
    names = [p.stem for p in paths]
    split_names = [split_on_underscore(name) for name in names]

    labels = [
        f'{nlist[2].replace("instrument(", "").replace(")", "")}'.replace("_", "+")
        for nlist in split_names
    ]

    dfs = []
    for i, (res, p) in enumerate(zip(results, paths)):
        par_names = [n[0].decode("utf-8") for n in res['Optimizer']['fit_parameter_names']]
        trace = res['Output']['Solutions']['solution0']['tracedata'][()]
        weights = res['Output']['Solutions']['solution0']['weights'][()]
        name = names[i]
        label = labels[i]

        df = pd.DataFrame(
            trace, columns=par_names
        )

        df["weights"] = weights
        df["name"] = name
        df["label"] = label

        dfs.append(df)

    return merge_data_sets(dfs)

def corners_seaborn(results, paths, **kwargs):
    df = make_struct_df(results, paths)

    dont_plot = ["name", "weights"]

    g = sns.pairplot(
        data=df[[c for c in df.columns if c not in dont_plot]],
        hue="label",
        corner=True,
        kind="kde",  # "scatter",
        diag_kind="hist",
        plot_kws={
            # "s": 1.,
            # "alpha":df["weights"]/df["weights"].max()
            "weights": df["weights"],
            "levels": 5, # [0.63, 0.97, 0.99, 0.999],
        },
        diag_kws={
            "bins": 25,
            "weights": df["weights"],
        }
    )

    g.map_lower(sns.scatterplot, s=1., marker="+", alpha=0.3)
    g.add_legend(title="Instrument", adjust_subtitles=True)
    g.fig.suptitle(f"{str(Path(paths[0]).parent.parent.stem)} (sns corner)")


def plot_corners_sns(path=None):
    results, paths = get_target_results(path=path)

    corners_seaborn(results, paths)
    plt.gcf()
    filename = Path(path) / "sns_corner.png"
    plt.savefig(filename, dpi=300)

if __name__ == "__main__":
    _dirs = [
        "WASP-19b",
        "WASP-17b",
        "WASP-12b",
        "HD-189733b",
        "HAT-P-26b",
        "HAT-P-12b",
        "HAT-P-1b",
        "WASP-39b",
        # "WASP-121b",
    ]

    dirs = [WDIR / "data/synthetic_spectra" / d for d in _dirs]

    for d in dirs:
        try:
            plot_corners_sns(d)
            pass
        except KeyError:
            print(f"Not plotting corner for {d.stem}")

        plt.close()

        try:
            plot_spectra(d)
        except KeyError:
            print(f"Not plotting spectrum for {d.stem}")

        plt.close()