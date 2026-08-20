"""
Meridian, a life expectancy estimator: Streamlit app.

Mirrors predict_life_expectancy.py: three statsmodels OLS models (least
information, coarse and elaborate), chosen by a nested consent question, fed by
a short form of plain-English inputs.

Educational project (Digital Futures Data Academy). Not affiliated with or
endorsed by the World Health Organization.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse

# --- MODELS ---
# predict_life_expectancy.py is the backend: it holds the team's model specs,
# trains all three, and provides the console version of the same prediction.
from predict_life_expectancy import MODELS, TRAINED, DATA, REGIONS, evaluate

# cache key: any change to the backend file invalidates the cached models and metrics
import hashlib
BACKEND_SIG = hashlib.md5(open(os.path.join(os.path.dirname(__file__),
    "predict_life_expectancy.py"), "rb").read()).hexdigest()

FORM_INTRO = {"minimal": "basic figures only, nothing sensitive",
              "coarse": "basic figures, plus a range for each sensitive measure",
              "full": "exact figures for the elaborate model"}

BAND_LABELS = {"Adult_mortality": "Adult mortality (per 1,000 population)",
               "Under_five_deaths": "Under-five deaths (per 1,000 live births)",
               "Incidents_HIV": "HIV incidence (per 1,000 population)"}

# numeric fields collected via st.number_input: column -> (label with units)
NUMERIC_FIELDS = {
    "Population_mln": "Population (millions)",
    "GDP_per_capita": "GDP per capita (USD)",
    "Schooling": "Average years of schooling",
    "Under_five_deaths": "Under-five deaths (per 1,000 live births)",
    "Adult_mortality": "Adult mortality (per 1,000 population)",
    "Infant_deaths": "Infant deaths (per 1,000 live births)",
    "BMI": "Average BMI",
    "Incidents_HIV": "HIV incidence (per 1,000 population)",
    "Hepatitis_B": "Hepatitis B immunization coverage (%)",
    "Polio": "Polio immunization coverage (%)",
    "Diphtheria": "Diphtheria immunization coverage (%)",
    "Measles": "Measles cases (per 1,000 population)",
    "Alcohol_consumption": "Alcohol consumption (litres per capita)",
    "Thinness_ten_nineteen_years": "Thinness, ages 10 to 19 (%)",
    "Thinness_five_nine_years": "Thinness, ages 5 to 9 (%)",
}


def field_label(col):
    """Form label for a column, falling back to a readable version of its name."""
    return NUMERIC_FIELDS.get(col, col.replace("_", " ").capitalize())

CONSENT_OPTIONS = ["Yes, use the full health dataset", "No, not exact figures"]
RANGE_OPTIONS = ["Yes, share ranges", "No, basic figures only"]

TARGET = 1.8      # the best existing model's score, set as the bar in the brief
CANDIDATES = 25   # candidate columns offered to feature selection
VIF_RMSE = 1.09   # notebook output: test RMSE of the variance-inflation input set


def form_plan(spec):
    """Which questions the form asks, derived from the model's own feature list."""
    feats = spec["features"]
    regions = [f for f in feats if f.startswith("Region_")]
    economy = [f for f in feats if f.startswith("Economy_status_")]
    bands = [c for c in spec["band_cols"] if c in feats]
    numbers = list(dict.fromkeys("GDP_per_capita" if f == "GDP_per_capita_log" else f for f in feats if f not in regions + economy + bands and f != "Year"))
    return {"regions": regions, "economy": economy, "bands": bands, "numbers": numbers}


@st.cache_resource
def load_data_and_models(backend_sig):
    df = DATA
    provenance = {"countries": int(df["Country"].nunique()), "records": int(len(df)),
                  "year_min": int(df["Year"].min()), "year_max": int(df["Year"].max())}
    stats = {col: {"median": float(df[col].median()), "min": float(df[col].min()), "max": float(df[col].max())}
             for col in df.select_dtypes(include="number").columns}

    fitted = TRAINED

    # test-set evaluation extras for the performance tab
    X = df.drop(columns=["Life_expectancy"])
    y = df["Life_expectancy"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    def design(rows, name):
        spec, state = MODELS[name], fitted[name]["state"]
        return sm.add_constant(spec["apply"](rows, state)[spec["features"]].astype(float),
                               has_constant="add")

    scatter = pd.DataFrame({"actual": y_test.values,
                            "predicted": fitted["full"]["model"].predict(
                                design(X_test, "full")).values})
    ex_names = ["Japan", "Tunisia", "Benin"]  # recognisable test-set countries spanning the range
    ex_rows = X_test[(X_test["Year"] == 2015) & (X_test["Country"].isin(ex_names))]
    examples = pd.DataFrame({
        "Country": ex_rows["Country"].values,
        "Actual": y_test.loc[ex_rows.index].round(1).values,
        "Elaborate": fitted["full"]["model"].predict(design(ex_rows, "full")).round(1).values,
        "Least": fitted["minimal"]["model"].predict(design(ex_rows, "minimal")).round(1).values,
    }).sort_values("Actual", ascending=False)

    # the "give or take" a single prediction carries: half the width of the 95%
    # prediction interval, averaged over the test set
    bounds = fitted["full"]["model"].get_prediction(design(X_test, "full")).summary_frame(alpha=0.05)
    give_take = float(((bounds["obs_ci_upper"] - bounds["obs_ci_lower"]) / 2).mean())

    return {
        "fitted": fitted,
        "stats": stats,
        "provenance": provenance,
        "scatter": scatter,
        "examples": examples,
        "give_take": give_take,
        "split": {"train": int(len(X) - len(X_test)), "test": int(len(X_test))},
    }


@st.cache_resource
def live_metrics(backend_sig):
    """Cached call into the backend's evaluate(); the app does no modelling itself."""
    return evaluate("full")



def inject_css():
    """Load the stylesheet from style.css (palette and type per WHO Brand Guidelines.md)."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def section(target, step, heading, intro=""):
    """Eyebrow + serif heading + optional short intro line. Used on the Estimator tab."""
    step_html = f'<p class="eyebrow">{step}</p>' if step else ""
    intro_html = f'<p class="sec-intro">{intro}</p>' if intro else ""
    target.markdown(f'<div class="sec">{step_html}<h2>{heading}</h2>{intro_html}</div>',
                    unsafe_allow_html=True)


def band(target, key, heading, caption="", first=False):
    """One label-rail row: heading and caption in the left rail, visual in the body.

    The hairline and the space above it ride on a separate rule element, so the
    first band on a tab simply skips it and sits straight under the tab bar.
    Returns the body column to write the visual into.
    """
    if not first:
        target.markdown('<div class="railband-rule"></div>', unsafe_allow_html=True)
    box = target.container(key=f"band_{key}")
    rail, body = box.columns([19, 47], gap="large")
    cap_html = f'<p class="cap">{caption}</p>' if caption else ""
    rail.markdown(f'<div class="rail"><h2>{heading}</h2>{cap_html}</div>', unsafe_allow_html=True)
    return body


def wideband(target, key, heading, caption=""):
    """A band whose visual needs the whole width: heading and caption run across the top."""
    target.markdown('<div class="railband-rule"></div>', unsafe_allow_html=True)
    box = target.container(key=f"band_{key}")
    cap_html = f'<p class="cap">{caption}</p>' if caption else ""
    box.markdown(f'<div class="rail wide"><h2>{heading}</h2>{cap_html}</div>', unsafe_allow_html=True)
    return box


def stat(label, value, unit="", sub="", muted=False):
    """One stat cell: micro label, display number, optional unit and sub-line."""
    unit_html = f'<span class="u">{unit}</span>' if unit else ""
    sub_html = f'<span class="sub">{sub}</span>' if sub else ""
    return (f'<div class="stat{" off" if muted else ""}"><span class="lab">{label}</span>'
            f'<span class="num">{value}{unit_html}</span>{sub_html}</div>')


def stat_row(target, cells, key, hero=False, lead=False):
    """A row of stats: one column per cell, grouped by whitespace alone.

    The band's own hairline is the only rule, so the cells carry no fill and no
    border. `lead` marks a row that opens a band body, `hero` the one oversized
    pair a tab is allowed.
    """
    prefix = "statrowhero" if hero else ("statrowlead" if lead else "statrow")
    with target.container(key=f"{prefix}_{key}"):
        for slot, cell in zip(st.columns(len(cells), gap="large"), cells):
            slot.markdown(cell, unsafe_allow_html=True)


def band_options(edges):
    """Readable range choices, built from however many bands the model uses."""
    fmt = (lambda v: f"{v:,.2f}") if edges[-2] < 10 else (lambda v: f"{v:,.0f}")
    n = len(edges) - 1
    out = [f"Lowest {n}th (under {fmt(edges[1])})"]
    out += [f"{fmt(edges[i])} to {fmt(edges[i + 1])}" for i in range(1, n - 1)]
    out.append(f"Highest {n}th (over {fmt(edges[-2])})")
    return out


st.set_page_config(page_title="Meridian · Life expectancy estimator", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
data = load_data_and_models(BACKEND_SIG)
metrics = live_metrics(BACKEND_SIG)
stats = data["stats"]
prov = data["provenance"]
fitted = data["fitted"]
split = data["split"]
err_full = fitted["full"]["rmse"]
err_min = fitted["minimal"]["rmse"]
err_crs = fitted["coarse"]["rmse"]
loco_pooled = metrics["loco_pooled"]
loco_n, loco_under_1 = metrics["loco_n"], metrics["loco_under_1"]
n_inputs_full = metrics["n_features"]
full_model = fitted["full"]["model"]
train_err = float(np.sqrt(full_model.ssr / full_model.nobs))  # training-set RMSE, for the candidate table
# the input the shipped model leans on least, named and scored from the fit itself
weakest_input = str(full_model.pvalues.drop("const").idxmax()).replace("_", " ")
weakest_p = float(full_model.pvalues.drop("const").max())
# how many inputs each consent tier collects, read off the model specs themselves
tier_inputs = {name: len(spec["features"]) for name, spec in MODELS.items()}
# the share of the least-information-to-elaborate gap that the coarse tier recovers
gap_closed = (err_min - err_crs) / (err_min - err_full)

# identity and provenance live in the sidebar; the main area is the three tabs
st.sidebar.markdown(
    '<div class="lockup">'
    '<svg width="44" height="44" viewBox="0 0 40 40" role="img" aria-label="Meridian mark: an arc with a point at its peak">'
    '<path d="M6 31 A 15.5 15.5 0 0 1 34 31" fill="none" stroke="#0093D5" stroke-width="3.5" stroke-linecap="round"/>'
    '<circle cx="20" cy="14.5" r="3.5" fill="#00205C"/></svg>'
    '<h1 class="title">Meridian</h1></div>'
    '<p class="sub">Life expectancy estimator</p>'
    '<div class="prov">' + "".join(
        f'<div class="prov-item"><span class="prov-k">{k}</span><span class="prov-v">{v}</span></div>'
        for k, v in [
            ("Data", f'WHO life expectancy dataset &middot; {prov["records"]:,} records &middot; '
                     f'{prov["countries"]} countries &middot; {prov["year_min"]}&ndash;{prov["year_max"]}'),
            ("Method", "Ordinary least squares regression"),
            ("Evaluation", "20% test set, never used to fit"),
        ]
    ) + "</div>",
    unsafe_allow_html=True,
)

tab_est, tab_meth, tab_perf = st.tabs(["Estimator", "Methodology", "Performance"])

tab_est.markdown(
    '<p class="lead">Estimates life expectancy for a country-year from its health and economic '
    'indicators. You decide how much data the model may use.</p>', unsafe_allow_html=True)

section(tab_est, "Step 1", "Choose how much data the model may use",
        "The elaborate model is about four times more accurate, but it asks for mortality and "
        "disease figures you may not want to share. The privacy-preserving option is selected by "
        "default.")

consent = tab_est.radio(
    "Do you consent to using advanced population data, which may include protected information, "
    "for better accuracy? (Y/N)",
    options=CONSENT_OPTIONS, index=1,
    captions=[
        f"Elaborate model &middot; typically within {err_full:.2f} years",
        "You choose what you can share next",
    ],
    help="Advanced population data means mortality, disease and health-system indicators "
         "(adult mortality, infant and under-five deaths, HIV incidence) plus region. Choosing No predicts "
         "from basic economic and demographic figures only. Full input lists are under "
         "“Model specification and data caveats” on the Methodology tab.",
    horizontal=True,
)

# --- TIER SELECTION ---
use_full = consent.startswith("Yes")

if use_full:
    tier = "full"
else:
    ranges = tab_est.radio(
        "Would you share ranges instead of exact figures?",
        options=RANGE_OPTIONS, index=1,
        captions=[f"Coarse model &middot; typically within {err_crs:.2f} years",
                  f"Least information model &middot; typically within {err_min:.2f} years"],
        help="A range means the band your country falls in, not the measured figure.",
        horizontal=True,
    )
    tier = "coarse" if ranges.startswith("Yes") else "minimal"

spec = MODELS[tier]
model_info = fitted[tier]
plan = form_plan(spec)
n_inputs = len(plan["numbers"]) + len(plan["bands"]) + 1 + bool(plan["economy"]) + bool(plan["regions"])

section(tab_est, "Step 2", "Enter the indicators",
        f"{n_inputs} entries: {FORM_INTRO[tier]}. Every field starts at the dataset median, so change only the ones you know.")

# Lithuania, 2015: a real test-set row, used to prefill the form on demand.
DEMO_COUNTRY = {"Year": 2015, "Economy status": "Developed", "Region": "European Union",
                "Adult_mortality": 165.2, "Under_five_deaths": 4.9, "Alcohol_consumption": 14.2,
                "BMI": 26.4, "Schooling": 13.0, "Incidents_HIV": 0.09, "Infant_deaths": 4.0,
                "Population_mln": 2.9, "GDP_per_capita": 14264.0}

if tab_est.button("Try a real country (Lithuania, 2015)"):
    st.session_state["in_year"] = DEMO_COUNTRY["Year"]
    st.session_state["in_economy"] = DEMO_COUNTRY["Economy status"]
    st.session_state["in_region"] = DEMO_COUNTRY["Region"]
    for colname in plan["numbers"]:
        s_col = stats[colname]
        val = DEMO_COUNTRY.get(colname, s_col["median"])
        st.session_state[f"in_{colname}"] = min(max(val, s_col["min"]), s_col["max"])
    for colname in plan["bands"]:
        edges = fitted[tier]["middles"][colname]["edges"]
        idx = int(pd.cut([DEMO_COUNTRY[colname]], edges, labels=False)[0])
        st.session_state[f"in_band_{colname}"] = band_options(edges)[idx]


# --- FORM ---
with tab_est.form("predict_form"):
    year = st.slider("Year", prov["year_min"], prov["year_max"], value=int(round(stats["Year"]["median"])), key="in_year")

    col_a, col_b = st.columns(2, gap="medium")
    economy = col_a.selectbox("Economy status", ["Developed", "Developing"], key="in_economy") if plan["economy"] else None
    region_help = ("The model separates some regions; the rest share a common baseline."
                   if plan["regions"] and len(plan["regions"]) < len(REGIONS) - 1 else None)
    region = col_b.selectbox("Region", REGIONS, help=region_help, key="in_region") if plan["regions"] else None

    number_answers = {}
    for i, colname in enumerate(plan["numbers"]):
        s_col = stats[colname]
        target = col_a if i % 2 == 0 else col_b
        number_answers[colname] = target.number_input(field_label(colname), min_value=s_col["min"], max_value=s_col["max"], value=s_col["median"], key=f"in_{colname}")

    band_answers = {}
    if plan["bands"]:
        st.markdown('<p class="sec-intro">Which range does your country fall in? No exact figure is entered or stored.</p>', unsafe_allow_html=True)
        slots = st.columns(len(plan["bands"]), gap="medium")
        for slot, colname in zip(slots, plan["bands"]):
            middle = model_info["middles"][colname]
            options = band_options(middle["edges"])
            chosen = options.index(slot.selectbox(field_label(colname), options, index=1, key=f"in_band_{colname}"))
            band_answers[colname] = middle["value"][chosen]

    submitted = st.form_submit_button("Predict life expectancy")

# --- PREDICTION ---
if submitted:
    row = {"Year": float(year), **number_answers, **band_answers}
    row.update({f: 1.0 if f.endswith(economy) else 0.0 for f in plan["economy"]})
    row.update({f: 1.0 if f == "Region_" + region else 0.0 for f in plan["regions"]})

    engineered = spec["apply"](pd.DataFrame([row]), model_info["state"])
    X_new = sm.add_constant(engineered[spec["features"]].astype(float), has_constant="add")
    interval = model_info["model"].get_prediction(X_new).summary_frame(alpha=0.05)
    prediction = float(interval["mean"].iloc[0])
    give_take = float(interval["obs_ci_upper"].iloc[0] - interval["obs_ci_lower"].iloc[0]) / 2

    section(tab_est, "Step 3", "Your estimate")
    tab_est.metric("Predicted life expectancy", f"{prediction:.1f} years")
    tab_est.caption(f"{spec['label']} · give or take {give_take:.1f} years (the range that holds 95% of "
                    f"true values) · typical miss {model_info['rmse']:.2f} years on test data")


# ============================ METHODOLOGY ============================
# Method only: how the model was built, what each consent tier collects, why
# the two mortality columns carry the fit, and how the input list was chosen.
# Every accuracy figure lives on the Performance tab instead.
meth = tab_meth.container(key="methbody")

# Band 1: the build pipeline, promoted out of the caveats expander.
body = band(meth, "pipeline", "How the model was built",
            "Clean the data, split it 80/20, engineer features on the training data only, fit, then "
            "score once on the 20% test set. Error is root mean squared error (RMSE).", first=True)
body.markdown(
    '<div class="strip">'
    '<span class="lab">Clean</span><span class="op">&#8594;</span>'
    '<span class="lab">Split 80/20</span><span class="op">&#8594;</span>'
    '<span class="lab">Engineer on training data</span><span class="op">&#8594;</span>'
    '<span class="lab">Fit</span><span class="op">&#8594;</span>'
    '<span class="lab">Score once on test data</span></div>', unsafe_allow_html=True)
stat_row(body, [
    stat("Fitted on", f"{split['train']:,}", "records",
         "Every transform, band edge and coefficient comes from this 80% alone"),
    stat("Scored on", f"{split['test']:,}", "records",
         "Read once, at the end, and never used to fit"),
], key="pipeline")

# Band 2: what the consent question actually costs the model in inputs.
body = band(meth, "tiers", "What each consent tier collects",
            "Mortality and disease figures are more sensitive than basic demographic ones, so the "
            "three tiers differ by how much of that health data they ask for.")
stat_row(body, [
    stat("Least information", f"{tier_inputs['minimal']}", "inputs",
         "Year, economy status, GDP per capita, schooling and region. No health data."),
    stat("Coarse", f"{tier_inputs['coarse']}", "inputs",
         "Plus decile positions for adult mortality, under-five deaths and HIV incidence."),
    stat("Elaborate", f"{tier_inputs['full']}", "inputs",
         "Plus those three as exact figures, and infant deaths, BMI and alcohol consumption."),
], key="tiers", lead=True)

# Band 3: why the tiers differ. The mortality pair is how WHO derives the answer.
body = band(meth, "circularity", "Most of the accuracy comes from two mortality columns",
            "Adult mortality and under-five deaths are the two indices WHO uses to calculate life "
            "expectancy in a model life table. Those two columns on their own reach 97.7% R&sup2;. "
            "With every mortality column removed the model reaches 90.2%. The other "
            f"{CANDIDATES - 2} columns add 0.22 years.")
body.markdown(
    '<div class="strip">'
    '<span class="lab">Adult mortality <sub>45</sub>q<sub>15</sub></span><span class="op">+</span>'
    '<span class="lab">Under-five deaths <sup>5</sup>q<sub>0</sub></span><span class="op">&#8594;</span>'
    '<span class="lab">WHO model life table</span><span class="op">&#8594;</span>'
    '<span class="lab">Life expectancy</span></div>', unsafe_allow_html=True)
# 97.7, 90.2, 1.23, 2.80 and the 0.22 above are notebook output (FINDINGS.md
# section 2, circularity). They describe cut-down models this app never fits,
# so they are literals rather than live figures.
stat_row(body, [
    stat("Those two columns only", "97.7", "% R&sup2;", "1.23 years off on an unseen country"),
    stat("All mortality removed", "90.2", "% R&sup2;", "2.80 years off on an unseen country", muted=True),
], key="circularity", hero=True)

# Band 4: how the shipped input list was chosen.
body = band(meth, "selection", "How the inputs were chosen",
            "Stepwise selection keeps an input if it improves the fit. Variance inflation removes an "
            "input if other inputs already predict it, which here removed infant deaths. The test "
            "data supports keeping infant deaths, so the stepwise set was shipped.")
stat_row(body, [
    stat("Stepwise &middot; shipped", f"{n_inputs_full}", f"of {CANDIDATES} inputs",
         "Keeps an input if it lowers error. Ran after the targeted transforms, which kept logged "
         "population and dropped hepatitis B"),
    stat("Variance inflation &middot; rejected", "15", f"of {CANDIDATES} inputs",
         f"Removes an input that repeats another. {VIF_RMSE - err_full:.2f} years worse on test data"),
], key="selection", lead=True)
body.markdown(
    f'<div class="split slim"><i style="width:{n_inputs_full / CANDIDATES:.1%}"></i></div>'
    f'<div class="splitk"><span>{n_inputs_full} inputs kept</span>'
    f'<span>{CANDIDATES - n_inputs_full} dropped</span></div>', unsafe_allow_html=True)

meth_exp = meth.container(key="methexp")

with meth_exp.expander("What the data showed"):
    st.markdown(f"""
| What we found | What we did | What it cost |
|---|---|---|
| Economy status recorded twice, as exact opposites (r = &minus;1.00) | Dropped one | Nothing |
| Infant and under-five deaths move together (r = 0.99) | Kept both | Nothing |
| Diphtheria and polio immunisation move together (r = 0.95) | Dropped both | Nothing measurable |
| The two child-thinness measures move together (r = 0.94) | Dropped both | Nothing measurable |
| Country is an identifier, {prov['countries']} of them | Excluded | Pooled error {loco_pooled:.2f} years with whole countries held out, against {err_full:.2f} on a random split |
| Measles reads 64 in 17% of records, coverage never exceeds 99 | Kept, flagged as filled values | Nothing measurable |
""")
    st.caption("Each drop was scored on test RMSE before it was applied.")

with meth_exp.expander("Model specification and data caveats"):
    st.markdown(f"""
**Candidate models**

| Candidate model | Inputs | RMSE, training | RMSE, test |
|---|---|---|---|
| Every candidate input | {CANDIDATES} | 1.06 years | 1.07 years |
| Dropped by correlation, by hand | 22 | 1.07 years | 1.09 years |
| Pruned by variance inflation (VIF) | 15 | 1.08 years | {VIF_RMSE:.2f} years |
| **Stepwise selection, used by this app** | **{n_inputs_full}** | **{train_err:.2f} years** | **{err_full:.2f} years** |

First three rows are notebook output; the last is computed live by this app.

Stepwise tests whether an input lowers error; VIF tests whether it repeats another. All
{n_inputs_full} shipped inputs are significant at p &lt; 0.01 (weakest {weakest_input},
p = {weakest_p:.3f}).
""")
    spec_cols = st.columns(3, gap="large")
    tier_specs = [
        ("minimal", "Least information", "Year, economy status, GDP per capita raw and logged, "
                                         "schooling, eight region flags. No health data."),
        ("coarse", "Coarse", "Year, population, schooling raw and logged, GDP per capita logged, eight region "
                             "flags, plus adult mortality, under-five deaths and HIV incidence as "
                             "training-set deciles."),
        ("full", "Elaborate", "Infant deaths logged, under-five deaths, adult mortality "
                              "square-rooted, population logged, economy status, schooling, BMI, "
                              "year, HIV incidence, alcohol consumption, six region flags."),
    ]
    for slot, (name, title, detail) in zip(spec_cols, tier_specs):
        slot.markdown(f'<span class="lab">{title}, {tier_inputs[name]} inputs</span>'
                      f'<p class="expcol">{detail}</p>', unsafe_allow_html=True)
    st.markdown("""
Measles and HIV incidence hold placeholder values in parts of the source data, so treat any single
coefficient as correlational only.

**Variation explained** on the Performance tab is R&sup2;, and most of that variation sits between
countries, which is why the leave-one-country-out check is stricter.
""")


# ============================ PERFORMANCE ============================
# Results only: what the three tiers cost in accuracy, how the shipped model
# stands against the benchmark and the naive baselines, and how it behaves on
# countries and records it never saw.
perf = tab_perf.container(key="perfbody")

# Band 1: the first beat arriving from Methodology, what each tier is worth.
body = band(perf, "ladder", "Accuracy by consent tier",
            "Typical error on test data for each of the three models. Coarse recovers "
            f"{gap_closed:.0%} of the difference between the least-information and elaborate tiers. "
            f"Only the elaborate model is under the {TARGET:.1f} year benchmark, the best existing "
            "model's score.", first=True)
stat_row(body, [
    stat("Least information", f"{err_min:.2f}", "yrs", "Economy status, GDP, schooling and region"),
    stat("Coarse", f"{err_crs:.2f}", "yrs", "Plus decile positions for three mortality measures"),
    stat("Elaborate", f"{err_full:.2f}", "yrs", "Plus exact health figures"),
], key="ladder", lead=True)

ladder = pd.DataFrame({
    "tier": ["Least information", "Coarse", "Elaborate"],
    "years": [err_min, err_crs, err_full],
    "shipped": ["no", "no", "yes"],
})
ladder["label"] = ladder["years"].map(lambda v: f"{v:.2f} years")
# direct value labels replace the axis entirely: three bars need no gridlines
ladder_bars = alt.Chart(ladder).mark_bar(height=30).encode(
    x=alt.X("years", axis=None, scale=alt.Scale(domain=[0, 6.6])),
    y=alt.Y("tier", sort=None, title=None),
    color=alt.condition(alt.datum.shipped == "yes", alt.value("#00205C"), alt.value("#0093D5")),
)
ladder_labels = alt.Chart(ladder).mark_text(align="left", dx=6, color="#00205C", fontSize=13).encode(
    x="years", y=alt.Y("tier", sort=None), text=alt.Text("label"))
bar_target = pd.DataFrame({"years": [TARGET], "label": [f"{TARGET:.1f} benchmark"]})
target_rule = alt.Chart(bar_target).mark_rule(color="#6E7C99", strokeDash=[4, 4]).encode(x="years")
target_label = alt.Chart(bar_target).mark_text(
    align="left", baseline="top", dx=6, dy=2, color="#5C6F94", fontSize=11, fontWeight=700,
).encode(x="years", y=alt.value(0), text="label")
with body.container(key="chart_ladder"):
    st.altair_chart(
        (ladder_bars + ladder_labels + target_rule + target_label).properties(height=200).configure_axis(
            labelColor="#45577D", titleColor="#00205C", domainColor="#B9C0D0",
            labelFontSize=12, titleFontSize=13, labelLimit=150, labelPadding=8,
        ).configure_view(strokeOpacity=0),
        use_container_width=True)

# Band 2: the shipped model against the bar it had to clear.
body = band(perf, "headline", f"{(TARGET - err_full) / TARGET:.0%} under the {TARGET:.1f} benchmark",
            f"The {TARGET:.1f} year benchmark is the best existing model's score. Average miss "
            f"{metrics['mae']:.2f} years, and {metrics['cv_mean']:.2f} &plusmn; "
            f"{metrics['cv_sd']:.2f} years across five refits of the training data.")
stat_row(body, [
    stat("Typical error", f"{err_full:.2f}", "yrs",
         "Root mean squared error on the 20% of records held back from fitting"),
    stat("Variation explained", f"{metrics['r2'] * 100:.1f}", "%", "R&sup2; on the same test data"),
    stat("Give or take", f"&plusmn;{data['give_take']:.1f}", "yrs",
         "95% prediction interval on a single estimate"),
], key="headline", lead=True)

# Band 3: the model against the naive baseline and the benchmark.
body = band(perf, "baselines", "Compared with simple baselines",
            "Predicting each country from its own historical average gives "
            f"{metrics['baseline_country_mean']:.2f} years of error, with no model fitted at all. "
            f"The model gives {err_full:.2f} years on the same records. Shorter is better.")
baseline = pd.DataFrame({
    "row": ["Country's own average", "Benchmark", "This model"],
    "years": [metrics["baseline_country_mean"], TARGET, err_full],
    "tone": ["naive", "target", "model"],
})
baseline["label"] = baseline["years"].map(lambda v: f"{v:.2f} years")
baseline_bars = alt.Chart(baseline).mark_bar(height=30).encode(
    x=alt.X("years", axis=None, scale=alt.Scale(domain=[0, 2.9])),
    y=alt.Y("row", sort=None, title=None),
    color=alt.Color("tone", legend=None, scale=alt.Scale(
        domain=["naive", "target", "model"], range=["#0093D5", "#B9C0D0", "#00205C"])),
)
baseline_labels = alt.Chart(baseline).mark_text(align="left", dx=6, color="#00205C", fontSize=13).encode(
    x="years", y=alt.Y("row", sort=None), text=alt.Text("label"))
with body.container(key="chartlead_baseline"):
    st.altair_chart(
        (baseline_bars + baseline_labels).properties(height=200).configure_axis(
            labelColor="#45577D", titleColor="#00205C", domainColor="#B9C0D0",
            labelFontSize=12, titleFontSize=13, labelLimit=170, labelPadding=8,
        ).configure_view(strokeOpacity=0),
        use_container_width=True)

# Band 4: the stricter test, whole countries removed one at a time.
body = band(perf, "loco", "Leave-one-country-out check",
            f"The model was refit {loco_n} times, each time with one country taken out of training "
            "and predicted from the rest. This is the leave-one-country-out (LOCO) check.")
stat_row(body, [
    stat("Average country", f"{metrics['loco_mean']:.2f}", "yrs"),
    stat("Median country", f"{metrics['loco_median']:.2f}", "yrs"),
    stat("Pooled", f"{loco_pooled:.2f}", "yrs"),
], key="loco", lead=True)
body.markdown(
    f'<div class="split"><i style="width:{loco_under_1 / loco_n:.1%}"></i></div>'
    f'<div class="splitk"><span>{loco_under_1} of {loco_n} countries under 1 year</span>'
    f'<span>{loco_n - loco_under_1} above</span></div>', unsafe_allow_html=True)

# Band 5: the parity plot, paired with three countries from the test set.
ex = data["examples"]
elab_off = (ex["Actual"] - ex["Elaborate"]).abs()
least_off = (ex["Actual"] - ex["Least"]).abs()
worst_country = ex.loc[least_off.idxmax(), "Country"]
box = wideband(perf, "parity", "Predicted against actual, test set",
               "Each point is one test-set country-year, and the dashed line is where predicted "
               "equals actual. The three countries beside the plot are 2015 records from the same "
               f"test set: the elaborate model is within {elab_off.max():.1f} years for all three, "
               f"while the least-information model is {least_off.max():.1f} years out for "
               f"{worst_country}.")
chart_col, cards_col = box.columns([460, 612], gap="large")
# equal width and height: a parity plot only reads correctly when the
# "perfect prediction" line sits at 45 degrees
TICKS = [40, 50, 60, 70, 80, 90]
axis_scale = alt.Scale(domain=[35, 90], nice=False)
diagonal = pd.DataFrame({"actual": [35, 90], "predicted": [35, 90]})
points = alt.Chart(data["scatter"]).mark_circle(size=30, opacity=0.55, color="#007EB4").encode(
    x=alt.X("actual", scale=axis_scale, title="Actual (years)", axis=alt.Axis(values=TICKS)),
    y=alt.Y("predicted", scale=axis_scale, title="Predicted (years)", axis=alt.Axis(values=TICKS)),
)
diag_line = alt.Chart(diagonal).mark_line(color="#00205C", strokeDash=[5, 4], size=1.5).encode(
    x=alt.X("actual", scale=axis_scale), y=alt.Y("predicted", scale=axis_scale))
chart_col.altair_chart(
    (points + diag_line).properties(width=330, height=330).configure_axis(
        labelColor="#45577D", titleColor="#00205C", gridColor="#EBEEF4",
        domainColor="#B9C0D0", labelFontSize=12, titleFontSize=13,
    ).configure_view(strokeOpacity=0),
    use_container_width=False)
# one row per country, straight from the live example predictions. The panel is
# the larger half of the pair: three countries, actual, and both outer tiers.
rows = "".join(
    f'<div class="ct-r"><span class="ct-c">{r["Country"]}</span>'
    f'<span class="ct-n">{r["Actual"]:.1f}</span>'
    f'<span class="ct-n">{r["Elaborate"]:.1f}<small>{abs(r["Actual"] - r["Elaborate"]):.1f} off</small></span>'
    f'<span class="ct-n q">{r["Least"]:.1f}<small>{abs(r["Actual"] - r["Least"]):.1f} off</small></span></div>'
    for _, r in ex.iterrows())
cards_col.markdown(
    '<div class="ct"><div class="ct-r h"><span>Test set, 2015</span><span>Actual</span>'
    '<span>Elaborate</span><span>Least information</span></div>'
    f'{rows}</div>', unsafe_allow_html=True)


st.markdown('<p class="foot">Independent educational project, not affiliated with or endorsed by '
            'the World Health Organization. Generative AI was used to build parts of this app; it '
            'accessed only public, country-level aggregate statistics, never sensitive or personal '
            'data.</p>', unsafe_allow_html=True)
