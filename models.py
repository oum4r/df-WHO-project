"""Model hand-off for the Meridian app. One block per model, one owner each.

Fill in YOUR block only and leave the others alone. Everything is copied from
your notebook: the final feature list your selection produced, plus your feature
engineering split in two (see the note at the bottom for why it is split).

Don't add split/fit/RMSE code, that is shared. Your functions are what the app
runs, so keep them as you wrote them.
"""
import numpy as np
import pandas as pd

import statsmodels.api as sm


# ===========================================================================
# MINIMAL  -  no sensitive data at all
# Owner: Bradley H.
# ===========================================================================
MINIMAL = {
    "label": "Minimal model",
    "features": ['Schooling', 'GDP_per_capita_log', 'Region_Asia',
       'Region_Central America and Caribbean', 'Region_South America', 'Year',
       'Region_Rest of Europe', 'Region_European Union', 'Region_Middle East',
       'Region_Oceania', 'Region_North America', 'Economy_status_Developed',
       'Economy_status_Developing', 'GDP_per_capita'],          # TODO your final column list, after engineering
    "band_cols": [],         # not a banded model, leave empty
}


def minimal_fit(train_df):
    return {}                # TODO anything learned from the training data, else {}


def minimal_apply(df, state):
    df = df.copy()
    # TODO your transforms (log, drops, ...)

    df['GDP_per_capita_log'] = np.log(df['GDP_per_capita'])

    df = df[MINIMAL['features']]

    return df.astype(float)


# ===========================================================================
# COARSE  -  sensitive measures shared as ranges, never exact figures
# Owner: Bradley H.
# ===========================================================================
COARSE = {
    "label": "Ranges model",
    "features": ['Year', 'Population_mln', 'Schooling', 'Adult_mortality',
       'Under_five_deaths', 'Incidents_HIV', 'Region_Asia',
       'Region_Central America and Caribbean', 'Region_European Union',
       'Region_Middle East', 'Region_North America', 'Region_Oceania',
       'Region_Rest of Europe', 'Region_South America', 'GDP_per_capita_log'],          # TODO your final list, band columns included
    "band_cols": ['Adult_mortality', 'Under_five_deaths', 'Incidents_HIV'],         # TODO raw columns you banded, e.g. ["Adult_mortality"]
}


def coarse_fit(train_df:pd.DataFrame) -> dict:
    state = {'edges':{}}

    for col in COARSE['band_cols']:
        edges = list(train_df[col].quantile([0,.25,.5,.75,1]).values)
        edges[0], edges[-1] = -np.inf, np.inf
        state['edges'][col] = edges

    return state                # TODO your quartile edges, from train_df only


def coarse_apply(df:pd.DataFrame, state:dict) -> pd.DataFrame:
    df = df.copy()
    # TODO your transforms. Bands MUST be guarded with `if col in df.columns:`
    # (see the note at the bottom) or the app will fail on a single input row.

    df['GDP_per_capita_log'] = np.log(df['GDP_per_capita'])
    df.drop('GDP_per_capita', axis=1, inplace=True)

    for col in COARSE['band_cols']:
        edges = state['edges'][col]
        df[col] = pd.cut(df[col], edges, labels=False)

    df = df[COARSE['features']]

    return df.astype(float)


# ===========================================================================
# FULL  -  exact figures, used only with consent
# Owner: Harun
# ===========================================================================
FULL = {
    "label": "Full model",
    # sel from the elaborate notebook: stepwise on p-values cut 25 columns to
    # these 15. Test RMSE 1.2174 vs 1.2160 all-features baseline.
    "features": [
        "Infant_deaths",
        "Adult_mortality",
        "Economy_status_Developing",
        "Region_Central America and Caribbean",
        "Region_South America",
        "Under_five_deaths",
        "GDP_per_capita_log",
        "Region_Oceania",
        "Region_European Union",
        "Schooling",
        "BMI",
        "Year",
        "Hepatitis_B",
        "Incidents_HIV",
        "Polio",
    ],
    "band_cols": [],         # not a banded model, leave empty
}


def full_fit(train_df):
    return {}                

def full_apply(df, state):
    df = df.copy()

    # add_gdp_log from the notebook. 
    if "GDP_per_capita" in df.columns:
        df["GDP_per_capita_log"] = np.log(df["GDP_per_capita"])
        df = df.drop(columns=["GDP_per_capita"])

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
