# PostRun

**Point-and-click statistical analysis for simulation output.**

Upload a CSV or Excel file of simulation results (one row per run or replication) and PostRun takes you from raw output to diagnosed models without writing code: exploratory plots, preprocessing, nine regression and classification models, and the full diagnostic suite.

## Features

- **Data prep:** missing-value imputation, outlier treatment (winsorize, clip, drop), log transforms, interaction terms, binning
- **Exploratory analysis:** histograms, box, violin, dot and stem-and-leaf plots, correlation heatmap, scatter matrix
- **Models:** linear, polynomial, robust (Huber/RANSAC), ridge, lasso, elastic net, logistic, random forest, gradient boosting
- **Tuning:** grid or random hyperparameter search with cross-validation
- **Diagnostics:** holdout R²/RMSE/MAE, F- and t-tests (where valid), VIF, Breusch-Pagan/White, Durbin-Watson, Ramsey RESET, Cook's distance, Q–Q plot, learning curves
- **Classification:** confusion matrix, ROC and precision-recall curves, F1, Brier score
- **ANOVA:** one-way ANOVA with Tukey HSD and η²

## Try it

Use the hosted app, or run it locally:

```bash
git clone https://github.com/<your-username>/postrun.git
cd postrun
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Requires **Python 3.10 or later**.

## Sample data

`sample_data.csv` is **synthetic** (500 rows), generated to look like supply-network simulation output:

| Column | Meaning |
|---|---|
| `fill_rate` | continuous target |
| `num_nodes` | network redundancy (1, 2, 4, 8) |
| `prob_disruption` | per-run disruption probability |
| `attrition_rate` | asset loss rate |
| `delay_hours` | transit delay |
| `policy` | categorical: baseline / prepositioned / surge |
| `replication` | run ID (no effect on the outcome) |
| `met_target` | binary target (fill_rate ≥ 0.5) |

In the app, click **Load example: supply-network simulation** in the sidebar to load this file with one click. Try `fill_rate` with Linear Regression, or `met_target` with Logistic Regression, and pick `policy` as the ANOVA factor. **When predicting `met_target`, remove `fill_rate` from the features.** `met_target` is computed from it, so leaving it in leaks the answer; the app warns when it detects this. The **Generate Sample Dataset** button gives a second, generic dataset.

## How categoricals are coded

Each categorical feature is reference-coded: a column with *k* levels becomes *k − 1* indicator columns, and the first level in alphabetical order is the reference. A coefficient such as `cat__policy_surge` is the difference between `surge` and the reference level (`baseline`), holding the other features constant.

## Data privacy

On the hosted version, uploaded files are processed on the hosting provider's servers. **Don't upload sensitive or proprietary data.** Run it locally for that.

## License

TBD
