import pandas as pd
from astroquery import mast
from astroquery.mast import Catalogs

def load_from_mast(which_kw=None):
    if which_kw is None:
        which_kw = {
            "instrument": "STIS",
            "telescope": "HST",
            "what": "spectrum",
        }

if __name__ == "__main__":
    ll_spectra = Catalogs.get_hsc_spectra()
    print(ll_spectra.colnames)