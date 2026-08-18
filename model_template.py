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
# If your engineering needs a value calculated across many rows (a median, a
# set of quantile edges), that value is what fit_engineer returns, and
# apply_engineer reads it back out of `state` rather than recalculating:
#
# def fit_engineer(train_df):
#     return {"some_column": <value calculated from train_df["some_column"]>}
#
# def apply_engineer(df, state):
#     df = df.copy()
#     df["some_column_derived"] = <use state["some_column"] on df["some_column"]>
#     return df
#
# The app predicts from a single row, so a quantile cannot be recalculated at
# predict time - that is the whole reason for the two-step split.
# ---------------------------------------------------------------------------
