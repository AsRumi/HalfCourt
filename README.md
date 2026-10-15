# HalfCourt

HalfCourt is an AI agent that takes NBA halftime box score data and produces two things: a quantitative prediction of the final game outcome from a trained ML model, and a natural language halftime commentary report written by an LLM that treats the model's prediction as fact.

The idea is that the ML model handles the actual prediction (win probability, margin) and the LLM's only job is to explain _why_ that outcome makes sense given the halftime stats. The LLM never contradicts the model.

---

## What it does

- Fetches per-player Q1+Q2 stats from the NBA API for any historical or live game
- Engineers ~43 halftime features (score state, shooting efficiency, pace, bench contribution, foul trouble, star player status, rest days, etc.)
- Runs a trained classifier (XGBoost, Logistic Regression, or MLP) to predict the winner and margin
- Extracts top-3 SHAP values to explain what drove the prediction
- Sends everything to an LLM (OpenAI / Anthropic / Gemini) to generate a broadcast-style halftime report
- Saves the report as both Markdown and JSON to `reports/`

---

## Relevant project structure

```
halfcourt/
-> config.yaml
-> .env                       # Place your Gemini API key here
-> requirements.txt
-> halfcourt.py               # CLI entry point
-> halfcourt_train.ipynb      # MLP training notebook (Google Colab)
-> data/
│   -> raw/                   # (not pushed since files were over 11000)
│   │   -> live/              # cached live game snapshots
│   -> processed/             # training parquet + csv
-> models/
│   -> saved/                 # serialized .joblib model files
-> reports/                   # previously generated reports you can take a look at
```

---

## Setup

**Python 3.10+ required.**

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Write `.env` and fill in your API keys:

```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

You only need the key for whichever provider is set in `config.yaml` under `llm.provider`. Currently it is set to `gemini`.

---

## Configuration

Everything tunable lives in `config.yaml`. The main things you might want to change:

```yaml
llm:
  provider: "gemini" # options: openai, anthropic, gemini
  model: "gemini-2.5-flash-lite"

model:
  type: "mlp" # default model for train/predict commands
```

---

## Running the full pipeline

There are four CLI commands. All are run from the project root.

### 1. Collect training data

This fetches halftime + final score data for the seasons listed in `config.yaml` and writes the processed dataset to `data/processed/halfcourt_training.parquet`.

```bash
python halfcourt.py collect
```

To collect specific seasons instead of the config defaults:

```bash
python halfcourt.py collect 2021-22 2022-23
```

Note: this hits the NBA API for every game in the specified seasons (~1,200 games per season). It takes 2-3 hours per season and caches aggressively to `data/raw/` so it can resume if interrupted.

### 2. Train a model

All three models were trained on Google Colab using the notebooks in the project root. Upload `data/processed/halfcourt_training.parquet` to Colab, run the notebook, and download the resulting `.joblib` file into `models/saved/`.

| Model               | Notebook                          | Output file                               |
| ------------------- | --------------------------------- | ----------------------------------------- |
| MLP (PyTorch)       | `halfcourt_train.ipynb`           | `models/saved/mlp.joblib`                 |
| XGBoost             | `train_xgboost.ipynb`             | `models/saved/xgboost.joblib`             |
| Logistic Regression | `train_logistic_regression.ipynb` | `models/saved/logistic_regression.joblib` |

To train locally:

```bash
python halfcourt.py train --model-type xgboost
python halfcourt.py train --model-type logistic_regression
python halfcourt.py train --model-type mlp
```

However, you do not have to train any model. All three models have been trained and saved to `models/saved`.

### 3. Predict on a completed historical game

Making a prediction requires the ID of that game. This ID can be found by going to nba.com -> games -> clicking on any game and copying the number from the link that looks like this: 0022300512

```bash
python halfcourt.py predict --game-id 0022300512 --model-type mlp
```

```bash
python halfcourt.py predict --game-id 0022300512 --model-type xgboost
python halfcourt.py predict --game-id 0022300512 --model-type logistic_regression
```

`--model-type` defaults to `mlp` if omitted. The game ID can be any completed NBA game. The report is written to `reports/`.

### 4. Predict on a live game (currently at halftime)

Getting the game ID for a live game follows the same process.

```bash
python halfcourt.py predict-live --game-id 0042500112 --model-type mlp
```

This uses a different data path (NBA CDN live endpoint instead of the stats API) so it works during the halftime break when the stats API returns incomplete data. The generated report will have a LIVE PREDICTION marker in the header.

The live snapshot is cached to `data/raw/live/{game_id}.json`. If you want to re-fetch during the same break (e.g. a player picked up another foul in the last minute), delete that file first.

### 5. Compare all three models

Trains and evaluates all three models back to back and prints the evaluation report for each:

```bash
python halfcourt.py compare
```

---

## Model notes

**MLP** is the default and has a dedicated margin prediction head trained together with the win probability head. It predicts both win probability and final point margin.

**XGBoost** predicts win probability. Margin is estimated from the win probability using a linear mapping so it will be less accurate than the MLP margin.

**Logistic Regression** is the baseline. It predicts win probability only. The generated report will note that margin prediction is not available when this model is used.

Expected accuracy on the test set is around 62-67% for all three models. NBA halftime data has real predictive signal but the second half is genuinely unpredictable, so this range is expected and on par with current models. One thing to note is that predictive models are not trained on halftime data (like HalfCourt) but on data from all quarters, therefore achieving higher accuracy than HalfCourt.

---

## Output

Reports are written to `reports/` as both `.md` and `.json`. They are grouped into one folder per game date, named `{away}_at_{home}`, e.g. `reports/2026-10-14/CHA_at_BOS.md`.

The markdown report contains:

- Model prediction summary (winner, win probability, margin, confidence, top SHAP features)
- LLM-generated commentary in four sections: Score & Momentum, Key Player Analysis, X-Factors / Foul Trouble, Second-Half Outlook

---

## Tech stack

- `nba_api` for historical data collection
- `pandas`, `numpy` for feature engineering
- `xgboost`, `scikit-learn`, `torch` for modeling
- `shap` for explainability
- `openai` / `anthropic` / `google-generativeai` for LLM commentary
- `joblib` for model serialization
