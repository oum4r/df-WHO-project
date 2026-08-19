"""Model hand-off for the Meridian app. One block per model, one owner each.

Fill in YOUR block only and leave the others alone. Everything is copied from
your notebook: the final feature list your selection produced, plus your feature
engineering split in two (see the note at the bottom for why it is split).

Don't add split/fit/RMSE code, that is shared. Your functions are what the app
runs, so keep them as you wrote them.
"""
import numpy as np
import pandas as pd


# ===========================================================================
# MINIMAL  -  no sensitive data at all
# Owner:
# ===========================================================================
MINIMAL = {
    "label": "Minimal model",
    "features": [],          # TODO your final column list, after engineering
    "band_cols": [],         # not a banded model, leave empty
}


def minimal_fit(train_df):
    return {}                # TODO anything learned from the training data, else {}


def minimal_apply(df, state):
    df = df.copy()
    # TODO your transforms (log, drops, ...)
    return df


# ===========================================================================
# COARSE  -  sensitive measures shared as ranges, never exact figures
# Owner:
# ===========================================================================
COARSE = {
    "label": "Ranges model",
    "features": [],          # TODO your final list, band columns included
    "band_cols": [],         # TODO raw columns you banded, e.g. ["Adult_mortality"]
}


def coarse_fit(train_df):
    return {}                # TODO your quartile edges, from train_df only


def coarse_apply(df, state):
    df = df.copy()
    # TODO your transforms. Bands MUST be guarded with `if col in df.columns:`
    # (see the note at the bottom) or the app will fail on a single input row.
    return df


# ===========================================================================
# FULL  -  exact figures, used only with consent
# Owner:
# ===========================================================================
FULL = {
    "label": "Full model",
    "features": [],          # TODO your final column list
    "band_cols": [],         # not a banded model, leave empty
}


def full_fit(train_df):
    return {}


def full_apply(df, state):
    df = df.copy()
    # TODO your transforms, or just return df unchanged
    return df


# ===========================================================================
# Assembled for the app. No need to edit below this line.
# ===========================================================================
MODELS = {
    "minimal": {**MINIMAL, "fit": minimal_fit, "apply": minimal_apply},
    "coarse": {**COARSE, "fit": coarse_fit, "apply": coarse_apply},
    "full": {**FULL, "fit": full_fit, "apply": full_apply},
}

# ---------------------------------------------------------------------------
# Why the engineering is split in two
#
# The app predicts from ONE row a user typed in, and you cannot work out a
# quantile from a single row. So anything calculated across many rows is worked
# out once, on the training data, and handed back for reuse:
#
#   def coarse_fit(train_df):
#       return {"some_column": <edges calculated from train_df["some_column"]>}
#
#   def coarse_apply(df, state):
#       df = df.copy()
#       for col in COARSE["band_cols"]:
#           if col in df.columns:        # <- REQUIRED
#               df[col + "_band"] = <use state[col] on df[col]>
#           # A row from the app carries the band already, because a band is all
#           # the user disclosed, so the raw column is absent and there is
#           # nothing to convert.
#       return df
#
# Rule of thumb: anything using .quantile / .median / .mean belongs in *_fit,
# everything else belongs in *_apply.
# ---------------------------------------------------------------------------
