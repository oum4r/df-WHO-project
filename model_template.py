"""Model hand-off for the Meridian app. One copy per model (minimal / elaborate).

Replace the four values below with YOUR final model; what's filled in is a
worked example. Your functions are what the app runs, so keep them as you wrote
them. Don't include split/fit/RMSE code, that's shared.

  MODEL          which model this is, and whose
  FEATURES       final column list your selection produced (copied, not re-derived)
  fit_engineer   anything learned from data, from TRAIN ONLY -> dict
  apply_engineer your transforms, applied to train, test, or one app input row
"""
import numpy as np
import pandas as pd

MODEL = "minimal - <your name>"          # or "elaborate - <your name>"

FEATURES = ["Year", "Economy_status_Developing", "Population_mln", "log_GDP", "Schooling"]


def fit_engineer(train_df):
    return {}


def apply_engineer(df, state):
    df = df.copy()
    df["log_GDP"] = np.log(df["GDP_per_capita"])
    return df


# ---------------------------------------------------------------------------
# Worked example if you use quartile bands. Edges come from TRAIN ONLY, which
# is what fit_engineer is for; apply_engineer then cuts any dataframe with them.
#
# BAND_COLS = ["Adult_mortality", "Under_five_deaths", "Incidents_HIV"]
#
# def fit_engineer(train_df):
#     edges = {}
#     for col in BAND_COLS:
#         e = list(train_df[col].quantile([0, .25, .5, .75, 1]).values)
#         e[0], e[-1] = -np.inf, np.inf
#         edges[col] = e
#     return edges
#
# def apply_engineer(df, state):
#     df = df.copy()
#     df["log_GDP"] = np.log(df["GDP_per_capita"])
#     for col in BAND_COLS:
#         df[col + "_band"] = pd.cut(df[col], state[col], labels=False)
#     return df
# ---------------------------------------------------------------------------
