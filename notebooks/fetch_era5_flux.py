"""Fetch the ERA5 flux field used by conservative_flux_ERA5.ipynb.

The notebook reads a local ``outputFLUX.grib`` that is not in the repository,
and the request that produced it is recorded nowhere -- in particular no date.
This script makes the request explicit and reproducible.

The date is a free choice: Appendix A demonstrates that ``sum(value * area)`` is
preserved to machine precision, which is an algebraic identity that holds for any
field. A different date changes the digits, not the conclusion. DATE below is
therefore stated rather than recovered, and any valid ERA5 date would serve.

Needs a CDS account with the ``reanalysis-era5-complete`` licence accepted, and
credentials in ~/.cdsapirc.
"""
import cdsapi

DATE = "2024-06-01"          # stated, not recovered -- see the docstring
GRID = "N256"                # reduced Gaussian, 348,528 points. MARS wants the
                             # Gaussian number here; "reduced_gg" is the GRIB
                             # gridType and is rejected as a GRID value.
OUT = "outputFLUX.grib"

# sshf = surface sensible heat flux, slhf = surface latent heat flux,
# ssr = surface net solar radiation. All accumulated (J/m^2) forecast fields.
PARAMS = "146.128/147.128/176.128"

request = {
    "class": "ea",
    "dataset": "reanalysis-era5-complete",
    "date": DATE,
    "expver": "1",
    "levtype": "sfc",
    "param": PARAMS,
    "step": "3/6",           # the notebook de-accumulates between these
    "stream": "enda",        # ensemble of data assimilation
    "number": "0/1/2/3/4/5/6/7/8/9",
    "time": "06:00:00",
    "type": "fc",            # forecast, not analysis -- accumulations need fc
    "grid": GRID,
}

if __name__ == "__main__":
    client = cdsapi.Client()
    print(f"requesting {PARAMS} for {DATE}, steps 3/6, 10 members, {GRID}")
    client.retrieve("reanalysis-era5-complete", request, OUT)
    import os
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
