"""
Meridian: life expectancy predictor. Single-file deliverable.

Three OLS models trained from the WHO life expectancy dataset, chosen by a
consent prompt: exact health figures, ranges only, or basic figures only.

Model specifications and their feature engineering were written by the team
(owners named per block below). This file trains them and exposes the console
prediction function. streamlit_app.py imports MODELS and TRAINED from here, so
the app and this function always run identically trained models.

Run: python predict_life_expectancy.py

Educational project. Not affiliated with or endorsed by the World Health
Organization.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from statsmodels.tools.eval_measures import rmse

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Life Expectancy Data.csv")

REGIONS = ["Africa", "Asia", "Central America and Caribbean", "European Union", "Middle East",
           "North America", "Oceania", "Rest of Europe", "South America"]
ALL_REGION_DUMMIES = [f"Region_{r}" for r in REGIONS if r != "Africa"]
BAND_COLS = ["Adult_mortality", "Under_five_deaths", "Incidents_HIV"]


# ===========================================================================
# MINIMAL  -  no sensitive data at all
# Owner: Bradley H.
# ===========================================================================
MINIMAL = {
    "label": "Least information model",
    "features": ['Schooling', 'GDP_per_capita_log', 'Region_Asia',
       'Region_Central America and Caribbean', 'Region_South America', 'Year',
       'Region_Rest of Europe', 'Region_European Union', 'Region_Middle East',
       'Region_Oceania', 'Region_North America',
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
    "label": "Coarse model",
    "features": ['Year', 'Population_mln', 'Schooling', 'Adult_mortality',
       'Under_five_deaths', 'Incidents_HIV', 'Region_Asia',
       'Region_Central America and Caribbean', 'Region_European Union',
       'Region_Middle East', 'Region_North America', 'Region_Oceania',
       'Region_Rest of Europe', 'Region_South America', 'GDP_per_capita_log'],          # TODO your final list, band columns included
    "band_cols": ['Adult_mortality', 'Under_five_deaths', 'Incidents_HIV'],         # TODO raw columns you banded, e.g. ["Adult_mortality"]
}


def coarse_fit(train_df:pd.DataFrame) -> dict:
    state = {'edges':{}}
    bands = 10

    for col in COARSE['band_cols']:
        edges = list(train_df[col].quantile([x/bands for x in range(bands+1)]).values)
        edges[0], edges[-1] = -np.inf, np.inf
        state['edges'][col] = edges

    return state                # TODO your band edges, from train_df only


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
    "label": "Elaborate model",
    # sel from the elaborate notebook: stepwise on p-values cut 25 columns to
    # these 16, selected by stepwise with Infant_deaths and Population_mln
    # logged and Adult_mortality square-rooted first (see full_apply).
    "features": [
        "Adult_mortality",
        "Under_five_deaths",
        "Alcohol_consumption",
        "Region_Central America and Caribbean",
        "Region_South America",
        "Economy_status_Developing",
        "BMI",
        "Schooling",
        "Incidents_HIV",
        "Region_North America",
        "Infant_deaths",
        "Region_Oceania",
        "Region_European Union",
        "Year",
        "Region_Asia",
        "Population_mln",
    ],
    "band_cols": [],         # not a banded model, leave empty
}


def full_fit(train_df):
    return {}                

def full_apply(df, state):
    df = df.copy()

    # Transformed in place rather than into _log / _sqrt columns: the prompt
    # loop below builds its questions from "features" and only special-cases
    # GDP_per_capita_log, so a renamed column would be asked for as a raw input.

    # right-skewed count (skew 1.10 -> -0.23 logged); needs a value > 0
    if "Infant_deaths" in df.columns:
        df["Infant_deaths"] = np.log(df["Infant_deaths"])

    
    if "Adult_mortality" in df.columns:
        df["Adult_mortality"] = np.sqrt(df["Adult_mortality"])

    # raw population is near-worthless to the model; logged it earns its place
    if "Population_mln" in df.columns:
        df["Population_mln"] = np.log(df["Population_mln"])

    return df


# ===========================================================================
# Assembled for the app. No need to edit below this line.
# ===========================================================================
MODELS = {
    "minimal": {**MINIMAL, "fit": minimal_fit, "apply": minimal_apply},
    "coarse": {**COARSE, "fit": coarse_fit, "apply": coarse_apply},
    "full": {**FULL, "fit": full_fit, "apply": full_apply},
}


# --- PROMPTS ---
PROMPTS = {
    "Year": "Year (2000-2015): ",
    "Economy_status_Developing": "Economy status (1 = Developing, 0 = Developed): ",
    "Population_mln": "Population (millions): ",
    "GDP_per_capita": "GDP per capita (USD): ",
    "Schooling": "Average years of schooling: ",
    "Infant_deaths": "Infant deaths (per 1,000 live births): ",
    "Under_five_deaths": "Under-five deaths (per 1,000 live births): ",
    "Alcohol_consumption": "Alcohol consumption (litres per capita): ",
    "Adult_mortality": "Adult mortality (per 1,000 population): ",
    "BMI": "Average BMI: ",
    "Incidents_HIV": "HIV incidence (per 1,000 population): ",
    "Hepatitis_B": "Hepatitis B immunization coverage (%): ",
    "Polio": "Polio immunization coverage (%): ",
}

# logged downstream, so these must be collected strictly above zero
LOG_INPUTS = {"GDP_per_capita", "Infant_deaths", "Population_mln"}


# --- TRAINING ---
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().replace(" ", "_").replace("/", "_") for c in df.columns]
    return pd.get_dummies(df, columns=["Region"], drop_first=True)


def fit_model(spec, df):
    """Split, learn state on train only, engineer both sides, fit, score."""
    X = df.drop(columns=["Life_expectancy"])
    y = df["Life_expectancy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    state = spec["fit"](X_train)
    train_fe = sm.add_constant(spec["apply"](X_train, state)[spec["features"]].astype(float), has_constant="add")
    test_fe = sm.add_constant(spec["apply"](X_test, state)[spec["features"]].astype(float), has_constant="add")
    model = sm.OLS(y_train, train_fe).fit()

    # middle value of each band: sent instead of a band number so the model's
    # own pd.cut lands it back in the band the user picked
    middles = {}
    for col in spec["band_cols"]:
        edges = state["edges"][col]
        band = pd.cut(X_train[col], edges, labels=False)
        middles[col] = {"edges": edges, "value": X_train.groupby(band)[col].median().to_dict()}

    return {"model": model, "state": state, "rmse": rmse(y_test, model.predict(test_fe)), "middles": middles}


DATA = load_data()
TRAINED = {name: fit_model(spec, DATA) for name, spec in MODELS.items()}


def evaluate(name="full"):
    """Performance figures for one model, computed on demand.

    Not run on import: the leave-one-country-out pass refits 179 times, which
    is fine for a cached call from the app but would slow the console script.
    """
    spec, trained = MODELS[name], TRAINED[name]
    X = DATA.drop(columns=["Life_expectancy"])
    y = DATA["Life_expectancy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def design(rows, state):
        return sm.add_constant(spec["apply"](rows, state)[spec["features"]].astype(float),
                               has_constant="add")

    pred = trained["model"].predict(design(X_test, trained["state"]))

    cv = []
    for tr, va in KFold(5, shuffle=True, random_state=42).split(X_train):
        a, b = X_train.iloc[tr], X_train.iloc[va]
        st = spec["fit"](a)
        m = sm.OLS(y_train.iloc[tr], design(a, st)).fit()
        cv.append(rmse(y_train.iloc[va], m.predict(design(b, st))))

    # loco holds one RMSE per country; all_sq holds every squared error, so the
    # pooled figure is like for like with the random-split rmse above
    loco, all_sq = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=DATA["Country"]):
        a, b = X.iloc[tr], X.iloc[te]
        st = spec["fit"](a)
        m = sm.OLS(y.iloc[tr], design(a, st)).fit()
        held = m.predict(design(b, st))
        all_sq.extend(np.square(y.iloc[te] - held))
        loco.append(rmse(y.iloc[te], held))
    loco = np.array(loco)

    country_mean = y_train.groupby(X_train["Country"]).mean()
    naive_rmse = rmse(y_test, X_test["Country"].map(country_mean))

    return {"rmse": trained["rmse"], "r2": r2_score(y_test, pred),
            "baseline_country_mean": float(naive_rmse),
            "mae": mean_absolute_error(y_test, pred),
            "cv_mean": float(np.mean(cv)), "cv_sd": float(np.std(cv)),
            "loco_mean": float(loco.mean()), "loco_median": float(np.median(loco)),
            "loco_pooled": float(np.sqrt(np.mean(all_sq))),
            "loco_under_1": int((loco < 1.0).sum()), "loco_n": len(loco),
            "n_features": len(spec["features"])}


# --- CONSOLE PREDICTION ---
def _ask_float(prompt, positive=False):
    """A number, re-asked until it parses (and is above zero when logged later)."""
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("  Please enter a number.")
            continue
        if positive and value <= 0:
            print("  Please enter a number above zero.")
            continue
        return value


def _ask_choice(prompt, n):
    """A whole number from 1 to n, re-asked until valid."""
    while True:
        try:
            choice = int(input(prompt))
        except ValueError:
            choice = 0
        if 1 <= choice <= n:
            return choice
        print(f"  Please enter a number from 1 to {n}.")


def _ask_yes_no(prompt):
    """Y/N in any casing, re-asked until valid."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer Y or N.")


def _ask_region():
    """A region name matched case-insensitively, returned in its canonical casing."""
    canonical = {r.lower(): r for r in REGIONS}
    while True:
        answer = input(f"\nRegion ({', '.join(REGIONS)}): ").strip().lower()
        if answer in canonical:
            return canonical[answer]
        print("  Please enter one of the regions listed.")


def _ask_band(col, middle):
    """Ask which band the country falls in, and return a value inside it."""
    e = middle["edges"]
    n = len(e) - 1
    fmt = (lambda v: f"{v:,.2f}") if e[-2] < 10 else (lambda v: f"{v:,.0f}")
    print(f"\n{PROMPTS[col].strip(': ')}, which band?")
    print(f"  1  Lowest {n}th (under {fmt(e[1])})")
    for i in range(1, n - 1):
        print(f"  {i + 1}  {fmt(e[i])} to {fmt(e[i + 1])}")
    print(f"  {n}  Highest {n}th (over {fmt(e[-2])})")
    return middle["value"][_ask_choice(f"  Choose 1-{n}: ", n) - 1]


def predict_life_expectancy():
    if _ask_yes_no("Do you consent to using advanced population data, which may include "
                   "protected information, for better accuracy? (Y/N)"):
        tier = "full"
    else:
        ranges = _ask_yes_no("Would you share ranges instead of exact figures? (Y/N)")
        tier = "coarse" if ranges else "minimal"

    spec, trained = MODELS[tier], TRAINED[tier]
    print(f"\nUsing the {spec['label'].lower()}.")

    row = {}
    for col in spec["features"]:
        if col.startswith("Region_") or col == "GDP_per_capita_log":
            continue
        if col in spec["band_cols"]:
            row[col] = _ask_band(col, trained["middles"][col])
        else:
            row[col] = _ask_float(PROMPTS[col], positive=col in LOG_INPUTS)

    # only the models that log GDP need it, and only if the loop above missed it
    if "GDP_per_capita_log" in spec["features"] and "GDP_per_capita" not in row:
        row["GDP_per_capita"] = _ask_float(PROMPTS["GDP_per_capita"], positive=True)

    region = _ask_region()
    for dummy in ALL_REGION_DUMMIES:
        row[dummy] = 1.0 if dummy == f"Region_{region}" else 0.0

    engineered = spec["apply"](pd.DataFrame([row]), trained["state"])
    X_new = sm.add_constant(engineered[spec["features"]].astype(float), has_constant="add")
    prediction = float(trained["model"].predict(X_new).iloc[0])

    print(f"\nPredicted life expectancy: {prediction:.1f} years")
    print(f"The true value is typically within {trained['rmse']:.2f} years of this estimate.")
    return prediction


if __name__ == "__main__":
    predict_life_expectancy()
