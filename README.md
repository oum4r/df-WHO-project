# Meridian: WHO life expectancy estimator

This predicts life expectancy for a country and year, using published health
and economic indicators. Built for the Digital Futures Data Academy WHO
project.

Some states won't consent to sharing exact health figures, so the estimator
works at three levels of disclosure and shows what each one costs in
accuracy.

| Tier | What a state shares | Typical error |
|---|---|---|
| Full | exact health figures | 1.06 years |
| Ranges | which tenth (decile band) it falls in, not the figure | 2.38 years |
| Minimal | basic economic and demographic figures only | 4.52 years |

The competitor benchmark to beat was 1.8 years.

## Running it

```
pip install -r requirements.txt
streamlit run streamlit_app.py     # the web app
python predict_life_expectancy.py  # the console version
```

## Files

| File | Purpose |
|---|---|
| `predict_life_expectancy.py` | model definitions and training, plus the console version of the predictor |
| `streamlit_app.py` | the web app, imports the trained models from the file above |
| `style.css` | app styling |
| `.streamlit/config.toml` | app theme |
| `Life Expectancy Data.csv` | WHO dataset, 2,864 records, 179 countries, 2000 to 2015 |
| `notebooks/WHO LEAST Notebook.ipynb` | analysis behind the minimal and ranges models |
| `notebooks/WHO ELABORATE Notebook.ipynb` | analysis behind the full model |
| `notebooks/WHO ELABORATE Notebook - Extended Stats.ipynb` | same analysis plus residual diagnostics and a held-out test report |
| `notebooks/WHO Base Notebook.ipynb` | shared exploratory analysis |

Model specifications live in `predict_life_expectancy.py`, one block per model
with its owner named. The app imports them, so both front ends always run
identically trained models.

## Method

Ordinary least squares. Data is split 80/20 with a fixed seed, and feature
engineering only ever sees the training set, so every quoted figure comes from
the held-out 20%. We used stepwise selection for feature selection, checked it
against variance inflation, and refit and scored each candidate set on data it
hadn't seen.

Independent educational project. Not affiliated with or endorsed by the World
Health Organization. The dataset is public, country-level aggregate statistics.
