import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
import re
import numpy.ma as ma
from numpy.random import default_rng

from copy import deepcopy
from corner import corner
from custom_corner_core import custom_corner_impl
from tqdm import tqdm
from itertools import combinations

from taurex.binning.fluxbinner import FluxBinner
from taurex.util.util import wnwidth_to_wlwidth

WDIR = Path().cwd().parent

from basic_plotting import get_target_results
from universal_divergence import estimate

def get_high_weight_subsample(path, size=500):
    results, paths = get_target_results(path=path)



    par_names = []
    traces = []
    weights = []
    truths = []

    rng = default_rng()
    subsamples = []

    for i, res in enumerate(results):
        par_name = np.array([n[0].decode("utf-8") for n in res['Optimizer']['fit_parameter_names']])
        par_name[np.argwhere(par_name == "C_O_ratio")] = "ace_co_ratio"
        trace = res['Output']['Solutions']['solution0']['tracedata'][()]
        weight = res['Output']['Solutions']['solution0']['weights'][()]

        truth = {}
        for i, name in enumerate(par_name):
            for group_k in res["ModelParameters"].keys():
                if isinstance(res["ModelParameters"][group_k], h5py.Dataset):
                    continue
                # print(res["ModelParameters"][group_k].keys())  # offsets are not in here???
                if name in res["ModelParameters"][group_k].keys():
                    assert truth.get(name) is None
                    truth[name] = res["ModelParameters"][group_k][name][()]

        subsample = np.zeros((size, len(par_name)))
        for i, name in enumerate(par_name):
            subsample[:, i] = rng.choice(trace[:, i], size=size, p=weight, replace=False, shuffle=False)

        par_names.append(par_name)
        traces.append(trace)
        weights.append(weight)
        truths.append(truth)
        subsamples.append(subsample)

    divergences_truth = []

    for i, (res, names) in tqdm(enumerate(zip(results, par_names)),
                                leave=False):
        divergence = {}
        for j, name in enumerate(names):
            # divergence[name] = estimate(traces[i][:, j].reshape((-1, 1)), truths[i][name] * np.ones((2, 1)))  # np.ones needs to be done to get the estimate to work
            try:
                divergence[name] = estimate(subsamples[i][:, j].reshape((-1, 1)),
                                            truths[i][name] * np.ones((2, 1)),
                                            n_jobs=6)
            except KeyError:
                pass
        divergences_truth.append(divergence)

    cross_divergences = {}
    for pair in tqdm(combinations(range(len(results)), 2),
                     leave=False):
        i, j = pair
        name = f"{Path(paths[i]).stem}-{Path(paths[j]).stem}"
        cross_divergences[name] = {}

        all_names = np.unique(np.concatenate((par_names[i], par_names[j])).flatten())

        for par_name in all_names:
            if not (par_name in par_names[i] and par_name in par_names[j]):
                continue
            pi = np.argwhere(par_names[i] == par_name)
            pj = np.argwhere(par_names[j] == par_name)

            cross_divergences[name][par_name] = estimate(
                subsamples[i][:, pi].reshape((-1, 1)),
                subsamples[j][:, pj].reshape((-1, 1)),
                n_jobs=6
            )

    return par_names, traces, weights, truths, subsamples, divergences_truth, cross_divergences


if __name__ == "__main__":
    _dirs = [
        "WASP-39b",
        "WASP-19b",
        "WASP-17b",
        "WASP-12b",
        "HD-189733b",
        "HAT-P-26b",
        "HAT-P-12b",
        "HAT-P-1b",
        # "WASP-39b",
        # "WASP-121b",
    ]

    dirs = [WDIR / "data/retrievals" / d for d in _dirs]

    # _dirs = [
    #     "WASP-39b",
    #     "WASP-121b",
    # ]
    #
    # dirs = [WDIR / "data/retrievals" / d for d in _dirs]

    dirs = [WDIR / "data/synthetic_spectra/WASP-39b/syn_offset/"]

    for d in dirs:
        get_high_weight_subsample(d)

        plt.close()
        raise NotImplementedError