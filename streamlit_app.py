"""
Meridian, a life expectancy estimator: Streamlit app.

Mirrors predict_life_expectancy.py: three statsmodels OLS models (minimal,
ranges/coarse, and full), chosen by a nested consent question, fed by a short
form of plain-English inputs.

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
              "full": "exact figures for the full model"}

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

TARGET = 1.8      # the rival contractor's benchmark, the bar this model has to clear
CANDIDATES = 25   # candidate columns offered to feature selection


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

    # held-out evaluation extras for the methodology tab
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
        "Actual 2015": y_test.loc[ex_rows.index].round(1).values,
        "Full model": fitted["full"]["model"].predict(design(ex_rows, "full")).round(1).values,
        "Minimal model": fitted["minimal"]["model"].predict(design(ex_rows, "minimal")).round(1).values,
    }).sort_values("Actual 2015", ascending=False)

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
    """Eyebrow + serif heading + optional short intro line."""
    step_html = f'<p class="eyebrow">{step}</p>' if step else ""
    intro_html = f'<p class="sec-intro">{intro}</p>' if intro else ""
    target.markdown(f'<div class="sec">{step_html}<h2>{heading}</h2>{intro_html}</div>',
                    unsafe_allow_html=True)


def stat(label, value, unit="", sub="", muted=False):
    """One stat cell: micro label, display number, optional unit and sub-line."""
    unit_html = f'<span class="u">{unit}</span>' if unit else ""
    sub_html = f'<span class="sub">{sub}</span>' if sub else ""
    return (f'<div class="stat{" off" if muted else ""}"><span class="lab">{label}</span>'
            f'<span class="num">{value}{unit_html}</span>{sub_html}</div>')


def stat_row(target, cells, key, hero=False):
    """A row of stats: one column per cell, grouped by whitespace and one hairline.

    The rule sits on the keyed container rather than on each cell, so it runs
    unbroken across the column gaps. No fills, no borders around the cells.
    """
    with target.container(key=f'{"statrowhero" if hero else "statrow"}_{key}'):
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
err_full = fitted["full"]["rmse"]
err_min = fitted["minimal"]["rmse"]
err_crs = fitted["coarse"]["rmse"]
loco_pooled = metrics["loco_pooled"]
full_model = fitted["full"]["model"]
train_err = float(np.sqrt(full_model.ssr / full_model.nobs))  # training-set RMSE, for the candidate table
# the input the shipped model leans on least, named and scored from the fit itself
weakest_input = str(full_model.pvalues.drop("const").idxmax()).replace("_", " ")
weakest_p = float(full_model.pvalues.drop("const").max())

# identity and provenance live in the sidebar; the main area is the two tabs
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
            ("Evaluation", "Held-out 20% test set, never used to fit"),
        ]
    ) + "</div>",
    unsafe_allow_html=True,
)

tab_est, tab_meth, tab_perf = st.tabs(["Estimator", "Methodology", "Performance"])

tab_est.markdown(
    '<p class="lead">Estimates life expectancy for a country-year from its health and economic '
    'indicators. You decide how much data the model may use.</p>', unsafe_allow_html=True)

section(tab_est, "Step 1", "Choose how much data the model may use",
        "The full model is about four times more accurate, but it asks for mortality and disease "
        "figures you may not want to share. The privacy-preserving option is selected by default.")

consent = tab_est.radio(
    "Do you consent to using advanced population data, which may include protected information, "
    "for better accuracy? (Y/N)",
    options=CONSENT_OPTIONS, index=1,
    captions=[
        f"Typically within {err_full:.2f} years",
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
        captions=[f"Typically within {err_crs:.2f} years", f"Typically within {err_min:.2f} years"],
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

# --- FORM ---
with tab_est.form("predict_form"):
    year = st.slider("Year", prov["year_min"], prov["year_max"], value=int(round(stats["Year"]["median"])))

    col_a, col_b = st.columns(2, gap="medium")
    economy = col_a.selectbox("Economy status", ["Developed", "Developing"]) if plan["economy"] else None
    region = col_b.selectbox("Region", REGIONS) if plan["regions"] else None
    if plan["regions"] and len(plan["regions"]) < len(REGIONS) - 1:
        col_b.caption("The model separates some regions; the rest share a common baseline.")

    number_answers = {}
    for i, colname in enumerate(plan["numbers"]):
        s_col = stats[colname]
        target = col_a if i % 2 == 0 else col_b
        number_answers[colname] = target.number_input(field_label(colname), min_value=s_col["min"], max_value=s_col["max"], value=s_col["median"])

    band_answers = {}
    if plan["bands"]:
        st.markdown('<p class="sec-intro">Which range does your country fall in? No exact figure is entered or stored.</p>', unsafe_allow_html=True)
        slots = st.columns(len(plan["bands"]), gap="medium")
        for slot, colname in zip(slots, plan["bands"]):
            middle = model_info["middles"][colname]
            options = band_options(middle["edges"])
            chosen = options.index(slot.selectbox(field_label(colname), options, index=1))
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
                    f"true values) · typical miss {model_info['rmse']:.2f} years on held-out test data")


# ============================ METHODOLOGY ============================
# Band 1: what each consent tier costs in accuracy.
section(tab_meth, "The trade", "What sharing more buys you")
stat_row(tab_meth, [
    stat("Minimal", f"{err_min:.2f}", "yrs off"),
    stat("Ranges", f"{err_crs:.2f}", "yrs off"),
    stat("Full", f"{err_full:.2f}", "yrs off"),
    stat("Target to beat", f"{TARGET:.2f}", "yrs"),
], key="tiers")

ladder = pd.DataFrame({
    "tier": ["Economy, GDP, schooling", "Plus mortality deciles", "Plus exact health figures"],
    "years": [err_min, err_crs, err_full],
    "shipped": ["no", "no", "yes"],
})
ladder["label"] = ladder["years"].map(lambda v: f"{v:.1f} years")
# direct value labels replace the axis entirely: three bars need no gridlines
ladder_bars = alt.Chart(ladder).mark_bar(height=26).encode(
    x=alt.X("years", axis=None, scale=alt.Scale(domain=[0, 6.6])),
    y=alt.Y("tier", sort=None, title=None),
    color=alt.condition(alt.datum.shipped == "yes", alt.value("#00205C"), alt.value("#0093D5")),
)
ladder_labels = alt.Chart(ladder).mark_text(align="left", dx=6, color="#00205C", fontSize=13).encode(
    x="years", y=alt.Y("tier", sort=None), text=alt.Text("label"))
bar_target = pd.DataFrame({"years": [TARGET], "label": [f"{TARGET:.1f} target"]})
target_rule = alt.Chart(bar_target).mark_rule(color="#6E7C99", strokeDash=[4, 4]).encode(x="years")
target_label = alt.Chart(bar_target).mark_text(
    align="left", baseline="top", dx=6, dy=2, color="#5C6F94", fontSize=11, fontWeight=700,
).encode(x="years", y=alt.value(0), text="label")
tab_meth.altair_chart(
    (ladder_bars + ladder_labels + target_rule + target_label).properties(height=190).configure_axis(
        labelColor="#45577D", titleColor="#00205C", domainColor="#B9C0D0",
        labelFontSize=12, titleFontSize=13, labelLimit=280,
    ).configure_view(strokeOpacity=0),
    use_container_width=True)
# the share of the minimal-to-full gap that ranges recover, from the live tier errors
gap_closed = (err_min - err_crs) / (err_min - err_full)
tab_meth.caption(f"Ranges close {gap_closed:.0%} of the gap. Only full clears the {TARGET:.1f} target.")

# Band 2: why the tiers differ. The mortality pair is how WHO derives the answer.
section(tab_meth, "Why the tiers differ", f"Two of {CANDIDATES} columns do 98% of the work")
tab_meth.markdown(
    '<div class="strip">'
    '<span class="lab">Adult mortality <sub>45</sub>q<sub>15</sub></span><span class="op">+</span>'
    '<span class="lab">Under-five deaths <sup>5</sup>q<sub>0</sub></span><span class="op">&#8594;</span>'
    '<span class="lab">WHO model life table</span><span class="op">&#8594;</span>'
    '<span class="lab">Life expectancy</span></div>', unsafe_allow_html=True)
# 97.7, 90.2, 1.23, 2.80 and the 0.22 below are notebook output (FINDINGS.md
# section 2, circularity). They describe cut-down models this app never fits,
# so they are literals rather than live figures.
stat_row(tab_meth, [
    stat("Those two columns alone", "97.7", "% explained", "1.23 years off, unseen country"),
    stat("Mortality removed", "90.2", "% explained", "2.80 years off, unseen country", muted=True),
], key="circularity", hero=True)
tab_meth.caption("These two inputs are what WHO uses to calculate life expectancy. The other "
                 f"{CANDIDATES - 2} columns add 0.22 years.")

# Band 3: how the shipped input list was chosen.
n_inputs_full = metrics["n_features"]
section(tab_meth, "", f"{n_inputs_full} inputs kept, {CANDIDATES - n_inputs_full} dropped")
stat_row(tab_meth, [
    stat("Stepwise &middot; shipped", f"{n_inputs_full}",
         f"of {CANDIDATES} &middot; {err_full:.2f} yrs", "Does it earn its place?"),
    stat("Variance inflation &middot; rejected", "15",
         f"of {CANDIDATES} &middot; 1.09 yrs", "Does it repeat another?"),
], key="selection")
tab_meth.caption("Stepwise made the picks after targeted transforms: logged population made the cut, "
                 "hepatitis B dropped out. VIF dropped infant deaths, which the held-out data keeps.")

with tab_meth.expander("What the data showed"):
    st.markdown("Every drop below started as an observation in the data, not a preference. None was "
                "acted on until the cost had been measured on held-out records.")
    st.markdown(f"""
| What we found | What we did | What it cost |
|---|---|---|
| Economy status recorded twice, as exact opposites (r = &minus;1.00) | Kept one. Two columns carrying one fact break the regression | Nothing, and unavoidable |
| Infant and under-five deaths move as one (r = 0.99): the second contains the first | Flagged as duplicates, then let selection decide. It kept both | Nothing: each still adds signal |
| Diphtheria and polio immunisation move together (r = 0.95) | Selection rejected both once mortality was in the model | Nothing measurable |
| The two child-thinness measures move together (r = 0.94) | Selection rejected both | Nothing measurable |
| Country is an identifier, {prov['countries']} of them | Excluded. The model would learn countries rather than relationships | Checked by holding out whole countries: pooled error {loco_pooled:.2f} vs {err_full:.2f} on a random split |
| Filled values: measles reads 64 in 17% of records, coverage never exceeds 99 | Kept, but flagged. Predictions are fine; those coefficients are not causes | Nothing measurable |
""")
    st.caption("Correlation alone was never the deciding tool. It only sees pairs, it does not say "
               "which of a pair to keep, and it cannot measure what a drop costs.")

with tab_meth.expander("Model specification and data caveats"):
    st.markdown(f"""
**How it was built.** Clean, split 80/20, engineer features on the training set only, fit, evaluate
once on held-out data. Every figure on these tabs comes from the 20% of records the models never saw.
Error is root mean squared error (RMSE), reproduced in the analysis notebook.

**Candidate models**

| Candidate model | Inputs | Typical error, training | Typical error, held-out |
|---|---|---|---|
| Every candidate input | {CANDIDATES} | 1.06 years | 1.07 years |
| Dropped by correlation, by hand | 22 | 1.07 years | 1.09 years |
| Pruned by variance inflation (VIF) | 15 | 1.08 years | 1.09 years |
| **Stepwise selection, used by this app** | **{n_inputs_full}** | **{train_err:.2f} years** | **{err_full:.2f} years** |

The three candidate rows are notebook output from the selection pass; the shipped row is computed
live by this app. Every input we removed was judged on held-out error rather than assumed.

**How the inputs were chosen.** Two standard tools were run on the training data, and they disagreed.

|  | Stepwise selection | Variance inflation (VIF) |
|---|---|---|
| **The question it asks** | Does this input earn its place? | Does this input repeat another? |
| **How it decides** | Adds the strongest, drops any that stop earning their place | Removes anything scoring above 5 |
| **Inputs kept** | {n_inputs_full} of {CANDIDATES} | 15 of {CANDIDATES} |
| **What it removed** | Measles, polio, diphtheria, hepatitis B, GDP per capita, both thinness measures, two region flags | Infant deaths, which the held-out data says is worth keeping |
| **Typical error** | **{err_full:.2f} years** | 1.09 years |
| **Verdict** | **Shipped** | Rejected |

All {n_inputs_full} inputs are significant at p &lt; 0.01, and the weakest, {weakest_input}, sits at
p = {weakest_p:.3f}, so nothing in the shipped model is borderline. Neither tool answers the question
that actually matters, which is whether removing an input makes predictions worse, so every candidate
set was refitted and scored on records the models had never seen.

**Why three models?** Some WHO indicators (mortality, disease incidence) are more sensitive
than basic demographic and economic figures. Rather than force an all-or-nothing choice, the
estimator offers three levels of disclosure and shows what each one costs in accuracy.

**Minimal model (13 inputs):** Year, Economy status, GDP per capita (raw and log-transformed),
Schooling, and eight region flags. No health data of any kind.

**Ranges model (15 inputs):** Year, Population, Schooling, GDP per capita (log-transformed),
eight region flags, plus adult mortality, under-five deaths and HIV incidence given as decile
positions (tenths of the training distribution) rather than measured values. Decile boundaries are
learned from the training data only.

**Full model ({n_inputs_full} inputs):** Infant deaths (log-transformed), Under-five deaths, Adult
mortality (square-root transformed), Population (log-transformed), Economy status, Schooling, BMI,
Year, HIV incidence, Alcohol consumption, and six region flags. Stepwise ran after the targeted
transforms, so logged population made the cut while hepatitis B immunization and GDP per capita
did not.

Some source columns (e.g. Measles, HIV incidence) contain pre-filled/placeholder values in parts
of the original dataset, so treat any single feature's effect on the prediction with caution:
these are correlational, not causal, estimates.

**Differences explained** on the Performance tab is R&sup2;, and most of the variation it rewards is
difference between countries, which is why the leave-one-country-out check is the harder test.
""")


# ============================ PERFORMANCE ============================
# Band 1: the headline, against the benchmark the model had to clear.
section(tab_perf, "Against the bar",
        f"Beats the {TARGET:.1f} target by {(TARGET - err_full) / TARGET:.0%}")
stat_row(tab_perf, [
    stat("Typical miss", f"{err_full:.2f}", "yrs"),
    stat("Country never seen", f"{metrics['loco_mean']:.2f}", "yrs"),
    stat("Differences explained", f"{metrics['r2'] * 100:.1f}", "%"),
    stat("Give or take", f"&plusmn;{data['give_take']:.1f}", "yrs"),
], key="headline")
tab_perf.caption(f"Average miss {metrics['mae']:.2f} years, and {metrics['cv_mean']:.2f} ± "
                 f"{metrics['cv_sd']:.2f} years across five refits of the training data.")

# Band 2: the model against the do-nothing baseline and the benchmark.
section(tab_perf, "", "Half the error of guessing a country's own average")
baseline = pd.DataFrame({
    "row": ["Country's own average", "Benchmark to get under", "This model"],
    "years": [metrics["baseline_country_mean"], TARGET, err_full],
    "tone": ["naive", "target", "model"],
})
baseline["label"] = baseline["years"].map(lambda v: f"{v:.2f} years")
baseline_bars = alt.Chart(baseline).mark_bar(height=26).encode(
    x=alt.X("years", axis=None, scale=alt.Scale(domain=[0, 2.9])),
    y=alt.Y("row", sort=None, title=None),
    color=alt.Color("tone", legend=None, scale=alt.Scale(
        domain=["naive", "target", "model"], range=["#0093D5", "#B9C0D0", "#00205C"])),
)
baseline_labels = alt.Chart(baseline).mark_text(align="left", dx=6, color="#00205C", fontSize=13).encode(
    x="years", y=alt.Y("row", sort=None), text=alt.Text("label"))
tab_perf.altair_chart(
    (baseline_bars + baseline_labels).properties(height=160).configure_axis(
        labelColor="#45577D", titleColor="#00205C", domainColor="#B9C0D0",
        labelFontSize=12, titleFontSize=13, labelLimit=280,
    ).configure_view(strokeOpacity=0),
    use_container_width=True)
tab_perf.caption("Shorter is better. Guessing each country from its own historical average is the "
                 "do-nothing baseline this has to beat.")

# Band 3: the stricter test, whole countries held out one at a time.
section(tab_perf, "The harder test", "Never seen the country, still under a year")
stat_row(tab_perf, [
    stat("Average country", f"{metrics['loco_mean']:.2f}", "yrs"),
    stat("Pooled", f"{loco_pooled:.2f}", "yrs"),
], key="loco", hero=True)
under_1, loco_n = metrics["loco_under_1"], metrics["loco_n"]
tab_perf.markdown(
    f'<div class="split"><i style="width:{under_1 / loco_n:.1%}"></i></div>'
    f'<div class="splitk"><span>{under_1} of {loco_n} countries under 1 year</span>'
    f'<span>{loco_n - under_1} above</span></div>', unsafe_allow_html=True)
tab_perf.caption(f"Refit {loco_n} times, one country removed each time. The median country scores "
                 f"{metrics['loco_median']:.2f} years.")

# Band 4: the parity plot, paired with three countries from the test set.
section(tab_perf, "", "Close at both ends of the range")
chart_col, cards_col = tab_perf.columns([3, 2], gap="large")
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
    (points + diag_line).properties(width=420, height=420).configure_axis(
        labelColor="#45577D", titleColor="#00205C", gridColor="#EBEEF4",
        domainColor="#B9C0D0", labelFontSize=12, titleFontSize=13,
    ).configure_view(strokeOpacity=0),
    use_container_width=False)
# one line per country, straight from the live example predictions
cards = "".join(
    f'<div class="excard"><span class="lab">{r["Country"]}</span>'
    f'<span class="exline">{r["Actual 2015"]:.1f} actual &middot; '
    f'<b>{r["Full model"]:.1f} full</b> &middot; {r["Minimal model"]:.1f} minimal</span></div>'
    for _, r in data["examples"].iterrows())
cards_col.markdown(f'<span class="lab hd">Test set, 2015</span>{cards}', unsafe_allow_html=True)
tab_perf.caption("One dot per held-out country-year, against the dashed line of perfect prediction. "
                 "Points hug the line across the whole range, so accuracy holds at both ends.")


st.markdown('<p class="foot">Independent educational project, not affiliated with or endorsed by '
            'the World Health Organization. Generative AI was used to build parts of this app; it '
            'accessed only public, country-level aggregate statistics, never sensitive or personal '
            'data.</p>', unsafe_allow_html=True)
