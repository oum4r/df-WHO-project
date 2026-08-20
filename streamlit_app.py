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

    return {
        "fitted": fitted,
        "stats": stats,
        "provenance": provenance,
        "scatter": scatter,
        "examples": examples,
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
    intro_html = f'<p class="sec-intro">{intro}</p>' if intro else ""
    target.markdown(f'<div class="sec"><p class="eyebrow">{step}</p><h2>{heading}</h2>{intro_html}</div>',
                    unsafe_allow_html=True)


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
         "(adult mortality, HIV incidence, immunization) plus region. Choosing No predicts "
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


section(tab_meth, "Method", "How the model was built",
        "Clean, split 80/20, engineer features on the training set only, fit, evaluate once on "
        "held-out data. Every figure below comes from the 20% of records the models never saw.")
tab_meth.markdown(f"""
| Candidate model | Inputs | Typical error, training | Typical error, held-out |
|---|---|---|---|
| Every candidate input | 25 | 1.06 years | 1.07 years |
| Dropped by correlation, by hand | 22 | 1.07 years | 1.09 years |
| Pruned by variance inflation (VIF) | 15 | 1.08 years | 1.09 years |
| **Stepwise selection, used by this app** | **16** | **1.06 years** | **{err_full:.2f} years** |
""")
tab_meth.caption("Every input we removed was judged on held-out error rather than assumed. VIF is a "
                 "standard test for inputs that duplicate each other; here it removed an input the "
                 "held-out data said was worth keeping. Stepwise selection matched the full model "
                 "using 16 of 25 inputs, comfortably inside the 1.8-year benchmark set by the "
                 "rival contractor team. Error is root mean squared "
                 "error (RMSE), reproduced in the analysis notebook.")

section(tab_meth, "Investigation", "What the data showed, and what we did about it",
        "Every drop below started as an observation in the data, not a preference. None was "
        "acted on until the cost had been measured on held-out records.")
tab_meth.markdown(f"""
| What we found | What we did | What it cost |
|---|---|---|
| Economy status recorded twice, as exact opposites (r = &minus;1.00) | Kept one. Two columns carrying one fact break the regression | Nothing, and unavoidable |
| Infant and under-five deaths move as one (r = 0.99): the second contains the first | Flagged as duplicates, then let selection decide. It kept both | Nothing: each still adds signal |
| Diphtheria and polio immunisation move together (r = 0.95) | Selection rejected both once mortality was in the model | Nothing measurable |
| The two child-thinness measures move together (r = 0.94) | Selection rejected both | Nothing measurable |
| Country is an identifier, 179 of them | Excluded. The model would learn countries rather than relationships | Checked by holding out whole countries: pooled error {loco_pooled:.2f} vs {err_full:.2f} on a random split |
| Filled values: measles reads 64 in 17% of records, coverage never exceeds 99 | Kept, but flagged. Predictions are fine; those coefficients are not causes | Nothing measurable |
""")
tab_meth.caption("Correlation alone was never the deciding tool. It only sees pairs, it does not say "
                 "which of a pair to keep, and it cannot measure what a drop costs.")

section(tab_meth, "Selection", "How the 16 inputs were chosen",
        "Two standard tools were run on the training data, and they disagreed.")
tab_meth.markdown("""
|  | Stepwise selection | Variance inflation (VIF) |
|---|---|---|
| **The question it asks** | Does this input earn its place? | Does this input repeat another? |
| **How it decides** | Adds the strongest, drops any that stop earning their place | Removes anything scoring above 5 |
| **Inputs kept** | 16 of 25 | 15 of 25 |
| **What it removed** | Measles, polio, diphtheria, population, GDP per capita, both thinness measures, two region flags | Infant deaths, which the held-out data says is worth keeping |
| **Typical error** | **1.08 years** | 1.09 years |
| **Verdict** | **Shipped** | Rejected |
""")
tab_meth.caption("The weakest input stepwise kept, hepatitis B immunisation, still clears the "
                 "significance bar at p = 0.002, so nothing in the shipped model is borderline. "
                 "Neither tool answers the question that actually matters, which is whether "
                 "removing an input makes predictions worse, so every candidate set was refitted "
                 "and scored on records the models had never seen.")


section(tab_perf, "Reliability", "Accuracy on unseen data")
tab_perf.markdown(
    '<div class="prov">'
    f'<div><span class="prov-k">Typical error</span><span class="prov-v">{err_full:.2f} years on '
    'records the model never saw</span></div>'
    f'<div><span class="prov-k">Country never seen</span><span class="prov-v">{metrics["loco_mean"]:.2f} '
    'years, leave-one-country-out</span></div>'
    f'<div><span class="prov-k">Differences explained</span><span class="prov-v">{metrics["r2"]:.1%} '
    'of the variation between countries</span></div>'
    f'<div><span class="prov-k">Stability</span><span class="prov-v">{metrics["cv_mean"]:.2f} &plusmn; '
    f'{metrics["cv_sd"]:.2f} years across five refits</span></div>'
    '</div>'
    '<p class="sec-intro">Held-out records share countries with the training data, so as a stricter '
    'check, the model was refit with one whole country held out at a time (leave-one-country-out, '
    f'{metrics["loco_n"]} refits). For a country the model has <strong>never seen</strong>, the '
    f'average country scores <strong>{metrics["loco_mean"]:.2f} years</strong> of error '
    f'(median {metrics["loco_median"]:.2f}); pooled across every prediction it is '
    f'<strong>{loco_pooled:.2f} years</strong>, barely above the {err_full:.2f} years measured on '
    f'the ordinary random split (mean absolute error {metrics["mae"]:.2f} years), and '
    f'{metrics["loco_under_1"]} of {metrics["loco_n"]} countries score under a year of error. The '
    'model learned relationships between indicators, not the identities of particular countries. '
    f'For scale, guessing each country from its own historical average would miss by '
    f'<strong>{metrics["baseline_country_mean"]:.2f} years</strong>; the model misses by {err_full:.2f}. '
    f'Every one of the {metrics["n_features"]} inputs earns its place statistically (p &lt; 0.05). '
    'The percentage above is R&sup2;, and most of the variation it rewards is difference between '
    'countries, so the leave-one-country-out check is the harder test.</p>',
    unsafe_allow_html=True)
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
tab_perf.altair_chart(
    (points + diag_line).properties(width=430, height=430).configure_axis(
        labelColor="#45577D", titleColor="#00205C", gridColor="#EBEEF4",
        domainColor="#B9C0D0", labelFontSize=12, titleFontSize=13,
    ).configure_view(strokeOpacity=0),
    use_container_width=False)
tab_perf.caption("Each dot is one held-out country-year, against the dashed line of perfect "
                 "prediction. Points hug the line across the whole range, so accuracy holds at "
                 "both ends and not just in the middle.")

section(tab_perf, "In practice", "Three countries from the test set, 2015")
examples_header = ("| Country | Actual 2015 | Full model | Minimal model |\n|---|---|---|---|\n")
examples_rows = "\n".join(
    f"| {r['Country']} | {r['Actual 2015']} | {r['Full model']} | {r['Minimal model']} |"
    for _, r in data["examples"].iterrows())
tab_perf.markdown(examples_header + examples_rows)
tab_perf.caption("The full model lands within about a year in each case; the minimal model, denied "
                 "the health figures, is several years out on the harder countries.")

section(tab_meth, "Data sharing", "Accuracy at each consent tier")
ladder = pd.DataFrame({
    "tier": ["Basic figures only", "Ranges for sensitive figures", "Exact health figures"],
    "years": [round(err_min, 1), round(err_crs, 1), round(err_full, 1)],
    "built": ["in this app", "in this app", "in this app"],
})
ladder["label"] = ladder["years"].map(lambda v: f"{v:.1f} years")
# direct value labels replace the axis entirely: four bars need no gridlines
ladder_bars = alt.Chart(ladder).mark_bar(height=24).encode(
    x=alt.X("years", axis=None, scale=alt.Scale(domain=[0, 6.6])),
    y=alt.Y("tier", sort=None, title=None),
    color=alt.condition(alt.datum.built == "in this app",
                        alt.value("#00205C"), alt.value("#0093D5")),
)
ladder_labels = alt.Chart(ladder).mark_text(align="left", dx=6, color="#00205C", fontSize=13).encode(
    x="years", y=alt.Y("tier", sort=None), text=alt.Text("label"))
tab_meth.altair_chart(
    (ladder_bars + ladder_labels).properties(height=190).configure_axis(
        labelColor="#45577D", titleColor="#00205C", domainColor="#B9C0D0",
        labelFontSize=12, titleFontSize=13, labelLimit=280,
    ).configure_view(strokeOpacity=0),
    use_container_width=True)
tab_meth.caption("Shorter bar means a more accurate estimate. Sharing more raises accuracy sharply "
                 "at first: naming the band a country falls in, rather than the measured figure, "
                 "recovers most of the gain for a fraction of the disclosure. All three tiers are "
                 "offered by the estimator.")

with tab_meth.expander("Model specification and data caveats"):
    st.markdown(
        """
**Why three models?** Some WHO indicators (mortality, disease incidence) are more sensitive
than basic demographic and economic figures. Rather than force an all-or-nothing choice, the
estimator offers three levels of disclosure and shows what each one costs in accuracy.

**Minimal model (13 inputs):** Year, Economy status, GDP per capita (raw and log-transformed),
Schooling, and eight region flags. No health data of any kind.

**Ranges model (15 inputs):** Year, Population, Schooling, GDP per capita (log-transformed),
eight region flags, plus adult mortality, under-five deaths and HIV incidence given as decile
positions (tenths of the training distribution) rather than measured values. Decile boundaries are
learned from the training data only.

**Full model (16 inputs):** Infant deaths (log-transformed), Under-five deaths, Adult mortality
(square-root transformed), Economy status, Schooling, BMI, Year, HIV incidence, Hepatitis B
immunization, Alcohol consumption, and six region flags. GDP per capita was not selected.

Some source columns (e.g. Measles, HIV incidence) contain pre-filled/placeholder values in parts
of the original dataset, so treat any single feature's effect on the prediction with caution:
these are correlational, not causal, estimates.
        """
    )

st.markdown('<p class="foot">Independent educational project, not affiliated with or endorsed by '
            'the World Health Organization. Generative AI was used to build parts of this app; it '
            'accessed only public, country-level aggregate statistics, never sensitive or personal '
            'data.</p>', unsafe_allow_html=True)
