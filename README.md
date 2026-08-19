# Meridian: WHO life expectancy estimator

Predicts life expectancy for a country-year from published health and economic
indicators. Built for the Digital Futures Data Academy WHO project.

The point of the project is consent: several states are unwilling to share
sensitive health figures, so the estimator offers three levels of disclosure and
shows what each one costs in accuracy.

| Tier | What a state shares | Typical error |
|---|---|---|
| Full | exact health figures | 1.22 years |
| Ranges | which quarter it falls in, not the figure | 2.90 years |
| Minimal | basic economic and demographic figures only | 4.52 years |

The competitor benchmark to beat was 1.8 years.

## Running it

```
pip install -r requirements.txt
streamlit run streamlit_app.py     # the web app
python predict_life_expectancy.py  # the console version
```

## Files

| File | |
|---|---|
| `predict_life_expectancy.py` | model specifications, training, and the console prediction function |
| `streamlit_app.py` | the web app: imports the trained models from the file above |
| `style.css` | app styling |
| `.streamlit/config.toml` | app theme |
| `Life Expectancy Data.csv` | WHO dataset, 2,864 records, 179 countries, 2000 to 2015 |
| `notebooks/WHO LEAST Notebook.ipynb` | analysis behind the minimal and ranges models |
| `notebooks/WHO ELABORATE Notebook.ipynb` | analysis behind the full model |
| `notebooks/WHO Base Notebook.ipynb` | shared exploratory analysis |

Model specifications live in `predict_life_expectancy.py`, one block per model
with its owner named. The app imports them, so both front ends always run
identically trained models.

## Method

Ordinary least squares. Data is split 80/20 with a fixed seed; feature
engineering is learned from the training set only; every quoted figure comes
from the held-out 20%. Feature selection was done with stepwise selection and
checked against variance inflation, with each candidate set refitted and scored
on unseen records.

Independent educational project. Not affiliated with or endorsed by the World
Health Organization. The dataset is public, country-level aggregate statistics.
