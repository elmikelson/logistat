import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc, classification_report,
    brier_score_loss, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold, learning_curve,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.base import clone, is_classifier as sk_is_classifier
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.formula.api import ols as smf_ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*ill-conditioned.*')

APP_NAME = "PostRun"
APP_VERSION = "1.0"

# ──────────────────────────────────────────────────────────────────────────────
# Page config & title
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"{APP_NAME}: Statistical Analysis for Simulation Output")
st.caption(f"v{APP_VERSION} · Explore, fit, and validate models with full diagnostic output.")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Uploads & Options
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
st.sidebar.caption("🔒 Files are processed on this app's host, not stored by PostRun. "
                   "Don't upload sensitive or proprietary data.")

st.sidebar.header("📊 Exploratory Plots")
plot_types = st.sidebar.multiselect(
    "Select Visualization Types:",
    ["Histogram", "Box Plot", "Violin Plot", "Dot Plot", "Stem-and-Leaf", "Correlation Heatmap", "Scatter Matrix"],
    default=["Histogram", "Box Plot"],
)

st.sidebar.header("🤖 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model:",
    [
        "Linear Regression", "Polynomial Regression", "Robust Regression",
        "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
        "Logistic Regression", "Random Forest", "Gradient Boosting",
    ],
)
poly_degree = st.sidebar.selectbox("Polynomial Degree:", [2, 3, 4, 5], index=0) if model_type == "Polynomial Regression" else 2
robust_type = st.sidebar.selectbox("Robust Type:", ["Huber", "RANSAC"], index=0) if model_type == "Robust Regression" else "Huber"
reg_alpha = st.sidebar.slider("Regularization alpha (α):", 0.0001, 100.0, 1.0, 0.01, format="%.4f") \
    if model_type in ["Ridge Regression", "Lasso Regression", "ElasticNet Regression"] else 1.0
l1_ratio = st.sidebar.slider("L1 ratio (ElasticNet):", 0.0, 1.0, 0.5, 0.05) \
    if model_type == "ElasticNet Regression" else 0.5
gb_n_estimators = st.sidebar.slider("GB: n_estimators:", 50, 500, 100, 50) \
    if model_type == "Gradient Boosting" else 100
gb_learning_rate = st.sidebar.slider("GB: learning rate:", 0.01, 0.5, 0.1, 0.01) \
    if model_type == "Gradient Boosting" else 0.1
gb_max_depth = st.sidebar.slider("GB: max depth:", 2, 8, 3) \
    if model_type == "Gradient Boosting" else 3

task_type = st.sidebar.radio(
    "Task Type:",
    ["Auto-detect", "Regression", "Classification"],
    help="For Random Forest and Gradient Boosting, Auto-detect infers task from the target variable.",
)

st.sidebar.header("🔧 Hyperparameter Tuning")
enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", False)
tuning_method = st.sidebar.selectbox("Search Method:", ["Random Search", "Grid Search"]) if enable_tuning else "Random Search"
tuning_n_iter = st.sidebar.slider("Random Search iterations:", 5, 50, 20) if (enable_tuning and tuning_method == "Random Search") else 20

st.sidebar.header("🧩 Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.header("📊 Statistical Tests")
statistical_tests = st.sidebar.multiselect(
    "Select Statistical Tests:",
    [
        "F-test (Overall Significance)",
        "T-test (Individual Coefficients)",
        "Normality Tests (Shapiro-Wilk)",
        "Normality Tests (Jarque-Bera)",
        "Cross Validation",
        "Residual Analysis",
    ],
    default=["F-test (Overall Significance)", "Cross Validation"],
)

st.sidebar.header("🧪 Toggles")
show_validation_tests = st.sidebar.checkbox("Show Model Validation Tests", True)
show_diagnostic_tests = st.sidebar.checkbox("Show Data Diagnostic Tests", True)
show_feature_analysis = st.sidebar.checkbox("Show Feature Analysis", True)

# Sample datasets. The choice is stored in session state so it survives reruns —
# every widget change reruns the script, and a one-shot flag would drop the data.
EXAMPLE_PATH = Path(__file__).parent / "sample_data.csv"
st.sidebar.header("🧪 Sample Data")
if st.sidebar.button("Load example: supply-network simulation",
                     help="500 synthetic simulation runs with a continuous and a binary outcome."):
    st.session_state.data_source = "example"
if st.sidebar.button("Generate Sample Dataset",
                     help="400 rows of generic synthetic data."):
    st.session_state.data_source = "generated"
if st.session_state.get("data_source") and uploaded_file is None:
    if st.sidebar.button("Clear sample data"):
        st.session_state.data_source = None

with st.sidebar.expander("ℹ️ About PostRun"):
    st.markdown(
        f"**PostRun v{APP_VERSION}**: point-and-click statistical analysis for simulation output.\n\n"
        "**Workflow:** load data → clean (missing values, outliers) → explore (plots) → "
        "pick a target and features → fit a model → read the diagnostics.\n\n"
        "**Categoricals** are reference-coded: a column with *k* levels becomes *k − 1* "
        "indicators, and the first level alphabetically is the reference.\n\n"
        "**Reading the tests:** the F-test and t-tests are computed on the training split, "
        "and only where the math is valid (OLS). Judge predictive performance by the "
        "holdout metrics and cross-validation.\n\n"
        "**Privacy:** uploads are processed on the host's servers. For sensitive data, run "
        "PostRun locally (`streamlit run streamlit_app.py`)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generate_sample_dataset(n_samples: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    feature_1 = rng.normal(50, 15, n_samples)
    feature_2 = rng.uniform(0, 100, n_samples)
    feature_3 = rng.exponential(2, n_samples)
    feature_4 = rng.integers(1, 10, n_samples)
    category_var = rng.choice(['A', 'B', 'C'], n_samples)
    target_continuous = (
        8 + 0.55*feature_1 - 0.28*feature_2 + 1.9*feature_3 + 4.8*feature_4 +
        1.2*(feature_1*feature_3/50) - 0.6*(feature_2 > 50).astype(int) +
        np.where(category_var == 'B', 6, 0) + np.where(category_var == 'C', -4, 0) +
        rng.normal(0, 9, n_samples)
    )
    z = (target_continuous - np.mean(target_continuous)) / np.std(target_continuous)
    prob = 1/(1+np.exp(-z))
    target_binary = rng.binomial(1, prob, n_samples)
    return pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'feature_4': feature_4,
        'category_var': category_var,
        'target_continuous': target_continuous,
        'target_binary': target_binary,
        'replicate': rng.integers(1, 6, n_samples)
    })


@st.cache_data
def load_uploaded_data(file_bytes: bytes, file_name: str, sheet: str | None = None) -> pd.DataFrame:
    """Load CSV or Excel from raw bytes; cache key is (bytes, name, sheet) so reruns are free."""
    if file_name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    return pd.read_csv(io.BytesIO(file_bytes))


def validate_data(df: pd.DataFrame) -> bool:
    if df.empty:
        st.error("Empty dataset uploaded")
        return False
    if len(df.columns) < 2:
        st.error("Need at least 2 columns (features + target)")
        return False
    return True


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Show missing value summary and let the user choose an imputation strategy per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return df

    st.subheader("🩹 Missing Value Imputation")
    st.write(f"**{len(missing)} column(s) contain missing values:**")
    miss_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Pct': (missing.values / len(df) * 100).round(1)
    })
    st.dataframe(miss_df, hide_index=True)

    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    cols = st.columns(min(3, len(missing)))
    for i, col in enumerate(missing.index):
        with cols[i % len(cols)]:
            is_numeric = col in num_cols
            options = (
                ["Mean", "Median", "Mode", "Constant", "Drop rows"]
                if is_numeric else
                ["Mode", "Constant", "Drop rows"]
            )
            strategy = st.selectbox(f"`{col}` ({missing[col]} missing)", options, key=f"impute_{col}")
            if strategy == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "Median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "Mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif strategy == "Constant":
                fill_val = st.text_input(f"Fill value for `{col}`", value="0", key=f"fill_{col}")
                try:
                    df[col] = df[col].fillna(float(fill_val) if is_numeric else fill_val)
                except ValueError:
                    df[col] = df[col].fillna(fill_val)
            elif strategy == "Drop rows":
                df = df.dropna(subset=[col])

    remaining = df.isnull().sum().sum()
    if remaining == 0:
        st.success("✅ No missing values remain after imputation")
    else:
        st.warning(f"⚠️ {remaining} missing value(s) still remain")
    return df


def select_variables(df: pd.DataFrame):
    st.subheader("🎯 Variable Selection")
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Select Target Variable:", df.columns.tolist())
    with c2:
        available_features = [c for c in df.columns if c != target_col]
        feature_cols = st.multiselect(
            "Select Feature Variables:", available_features,
            default=available_features[:min(6, len(available_features))]
        )
    if not feature_cols:
        st.warning("Please select at least one feature variable")
        return None, None, None, None
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, target_col, feature_cols


def build_preprocessor(df: pd.DataFrame, feature_cols):
    Xraw = df[feature_cols]
    num_cols = Xraw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    # drop="first": each categorical contributes (levels − 1) indicator columns, with the
    # first level in sorted order as the reference. Keeping a full dummy set would make
    # the indicators sum to one, which is exactly collinear with an intercept (the
    # dummy-variable trap) and makes the OLS t-tests, VIF, Breusch-Pagan, White, RESET
    # and influence diagnostics singular for any categorical with three or more levels.
    # handle_unknown="ignore" encodes an unseen level as all zeros, i.e. the reference.
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def sample_size_advice(model_type, y, p):
    """Model-aware sample size check."""
    n = int(len(y))
    ok = False
    msg = ""

    regression_types = {
        "Linear Regression", "Polynomial Regression", "Robust Regression",
        "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
    }
    if model_type in regression_types:
        needed = max(30, 10 * max(p, 1))
        ok = n >= needed
        msg = f"Regression guideline: need ≥ max(30, 10×p) with p={p} ⇒ ≥ {needed} rows."

    elif model_type == "Logistic Regression":
        y_series = pd.Series(y)
        min_class = int(y_series.value_counts(dropna=False).min()) if len(y_series) else 0
        epv = min_class / max(p, 1)
        ok = epv >= 10
        msg = f"Logistic guideline: EPV≥10. min(class count)={min_class}, p={p} ⇒ EPV={epv:.1f}."

    elif model_type in ["Random Forest", "Gradient Boosting"]:
        min_needed = 300
        ok = n >= min_needed
        msg = f"{model_type} heuristic: aim for ≥ {min_needed} rows (more is better)."

    else:
        ok = n >= 30
        msg = "Generic CLT check: n ≥ 30."

    return ok, n, msg


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers in numeric columns and let the user choose treatment per column."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df

    # Detect IQR-based outliers across all numeric columns
    outlier_counts = {}
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if n_out > 0:
            outlier_counts[col] = n_out

    if not outlier_counts:
        return df

    st.subheader("🔎 Outlier Treatment")
    out_summary = pd.DataFrame({
        'Column': list(outlier_counts.keys()),
        'Outliers (IQR)': list(outlier_counts.values()),
        'Pct': [round(v / len(df) * 100, 1) for v in outlier_counts.values()]
    })
    st.dataframe(out_summary, hide_index=True)

    df = df.copy()
    cols = st.columns(min(3, len(outlier_counts)))
    for i, col in enumerate(outlier_counts):
        with cols[i % len(cols)]:
            strategy = st.selectbox(
                f"`{col}` ({outlier_counts[col]} outliers)",
                ["Keep", "Winsorize (IQR)", "Clip (z-score 3σ)", "Drop rows"],
                key=f"outlier_{col}"
            )
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            if strategy == "Winsorize (IQR)":
                df[col] = df[col].clip(lower=lb, upper=ub)
                st.success(f"✅ Winsorized to [{lb:.2f}, {ub:.2f}]")
            elif strategy == "Clip (z-score 3σ)":
                mu, sigma = df[col].mean(), df[col].std()
                df[col] = df[col].clip(lower=mu - 3*sigma, upper=mu + 3*sigma)
                st.success(f"✅ Clipped to [{mu-3*sigma:.2f}, {mu+3*sigma:.2f}]")
            elif strategy == "Drop rows":
                before = len(df)
                df = df[(df[col] >= lb) & (df[col] <= ub)]
                st.success(f"✅ Dropped {before - len(df)} rows")
    return df


def apply_feature_engineering(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, list]:
    """Let users apply log transforms, interaction terms, and binning to numeric features."""
    num_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_feats:
        return df, feature_cols

    st.subheader("⚙️ Feature Engineering")
    df = df.copy()
    new_cols = []

    with st.expander("🔢 Log Transforms"):
        log_cols = st.multiselect(
            "Apply log(x+1) to:", num_feats,
            help="Useful for right-skewed distributions. Requires all values ≥ 0."
        )
        for col in log_cols:
            if (df[col] < 0).any():
                st.warning(f"⚠️ `{col}` has negative values — skipping log transform")
            else:
                new_name = f"log1p_{col}"
                df[new_name] = np.log1p(df[col])
                new_cols.append(new_name)
                st.success(f"✅ Created `{new_name}`")

    with st.expander("✖️ Interaction Terms"):
        if len(num_feats) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                feat_a = st.selectbox("Feature A:", num_feats, key="interact_a")
            with c2:
                feat_b = st.selectbox("Feature B:", [f for f in num_feats if f != feat_a], key="interact_b")
            if st.button("Add A × B interaction"):
                new_name = f"{feat_a}_x_{feat_b}"
                df[new_name] = df[feat_a] * df[feat_b]
                st.session_state.setdefault("interaction_cols", [])
                if new_name not in st.session_state["interaction_cols"]:
                    st.session_state["interaction_cols"].append(new_name)
                st.success(f"✅ Created `{new_name}`")
            for ic in st.session_state.get("interaction_cols", []):
                if ic not in df.columns:
                    parts = ic.split("_x_", 1)
                    if len(parts) == 2 and all(p in df.columns for p in parts):
                        df[ic] = df[parts[0]] * df[parts[1]]
                if ic in df.columns and ic not in new_cols:
                    new_cols.append(ic)
        else:
            st.info("Need at least 2 numeric features for interactions.")

    with st.expander("📦 Binning (Numeric → Categorical)"):
        bin_col = st.selectbox("Column to bin:", ["(none)"] + num_feats, key="bin_col")
        if bin_col != "(none)":
            n_bins = st.slider("Number of bins:", 2, 10, 4, key="n_bins")
            bin_labels = st.checkbox("Use bin labels (low/med/high style)", value=True, key="bin_labels")
            new_name = f"{bin_col}_binned"
            if bin_labels:
                label_list = [f"bin{i+1}" for i in range(n_bins)]
                df[new_name] = pd.cut(df[bin_col], bins=n_bins, labels=label_list)
            else:
                df[new_name] = pd.cut(df[bin_col], bins=n_bins)
            df[new_name] = df[new_name].astype(str)
            new_cols.append(new_name)
            st.success(f"✅ Created `{new_name}` with {n_bins} bins")
            st.dataframe(df[new_name].value_counts().reset_index(), hide_index=True)

    updated_features = feature_cols + [c for c in new_cols if c not in feature_cols]
    if new_cols:
        st.info(f"Added {len(new_cols)} engineered feature(s): {', '.join(new_cols)}")
    return df, updated_features


# ──────────────────────────────────────────────────────────────────────────────
# EDA Plot Functions
# ──────────────────────────────────────────────────────────────────────────────

def create_histograms(df, columns):
    n_cols = len(columns)
    n_subplot_cols = min(3, n_cols)
    n_subplot_rows = (n_cols + n_subplot_cols - 1) // n_subplot_cols
    if n_cols == 1:
        col = columns[0]
        fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=30, marginal="box")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [df[col].count(), df[col].mean(), df[col].std(), df[col].min(),
                      df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75), df[col].max()]
        })
        st.dataframe(stats_df, hide_index=True)
    else:
        fig = make_subplots(rows=n_subplot_rows, cols=n_subplot_cols,
                            subplot_titles=[f"Distribution of {c}" for c in columns])
        for i, c in enumerate(columns):
            r = (i // n_subplot_cols) + 1
            cc = (i % n_subplot_cols) + 1
            fig.add_trace(go.Histogram(x=df[c], name=c, showlegend=False, nbinsx=20), row=r, col=cc)
        fig.update_layout(height=300*n_subplot_rows, title="Data Distributions")
        st.plotly_chart(fig, width="stretch")


def create_box_plots(df, columns):
    if len(columns) == 1:
        col = columns[0]
        fig = px.box(df, y=col, title=f"Box Plot of {col}", points="all")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        out_n = df[(df[col] < lb) | (df[col] > ub)][col].shape[0]
        st.write(f"**{col} Outlier Analysis:** IQR={IQR:.3f}, Lower={lb:.3f}, Upper={ub:.3f}, Outliers={out_n}")
    else:
        melted = df[columns].melt(var_name='Variable', value_name='Value')
        fig = px.box(melted, x='Variable', y='Value', title="Comparative Box Plots", points="outliers")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")


def create_violin_plots(df, columns):
    if len(columns) == 1:
        col = columns[0]
        fig = px.violin(df, y=col, title=f"Violin Plot of {col}", box=True, points="outliers")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Mean", f"{df[col].mean():.3f}")
        with c2: st.metric("Median", f"{df[col].median():.3f}")
        with c3: st.metric("Std Dev", f"{df[col].std():.3f}")
        with c4: st.metric("IQR", f"{df[col].quantile(0.75) - df[col].quantile(0.25):.3f}")
    else:
        melted = df[columns].melt(var_name='Variable', value_name='Value')
        fig = px.violin(melted, x='Variable', y='Value', title="Comparative Violin Plots", box=True, points="outliers")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")


def create_dot_plots(df, columns):
    for col in columns:
        st.write(f"**Dot Plot: {col}**")
        fig = px.strip(df, y=col, title=f"Dot Plot of {col}")
        fig.update_traces(jitter=0.4)
        fig.update_layout(height=400, yaxis_title=col)
        st.plotly_chart(fig, width="stretch")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Mean", f"{df[col].mean():.3f}")
        with c2: st.metric("Median", f"{df[col].median():.3f}")
        with c3: st.metric("Std Dev", f"{df[col].std():.3f}")


def create_stem_leaf_plots(df, columns):
    for col in columns:
        st.write(f"**Stem-and-Leaf Plot: {col}**")
        data = df[col].dropna().sort_values()
        if len(data) == 0:
            st.warning(f"No data available for {col}")
            continue
        if len(data) > 200:
            st.info(f"Large dataset ({len(data)}). Showing sampled 200 for readability.")
            data = data.sample(200, random_state=random_state).sort_values()
        rng = data.max() - data.min()
        if rng == 0:
            st.write("All values equal:", data.iloc[0])
            continue
        abs_data = data.abs()
        signs = np.where(data >= 0, 1, -1)
        if rng < 100:
            raw_stems = (abs_data // 10).astype(int) * signs
            leaves = (abs_data % 10).astype(int)
            note = "(Stem=tens, Leaf=ones)"
        elif rng < 1000:
            raw_stems = (abs_data // 100).astype(int) * signs
            leaves = ((abs_data % 100)//10).astype(int)
            note = "(Stem=hundreds, Leaf=tens)"
        else:
            raw_stems = (abs_data // 1000).astype(int) * signs
            leaves = ((abs_data % 1000)//100).astype(int)
            note = "(Stem=thousands, Leaf=hundreds)"
        stem_leaf = {}
        for s, l in zip(raw_stems, leaves):
            stem_leaf.setdefault(s, []).append(l)
        st.text(f"Stem-and-Leaf Plot {note}\n" + "-"*40)
        lines = [f"{s:3d} | {''.join(str(x) for x in sorted(stem_leaf[s]))}" for s in sorted(stem_leaf)]
        st.text('\n'.join(lines))
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("N", len(data))
        with c2: st.metric("Min", f"{data.min():.2f}")
        with c3: st.metric("Max", f"{data.max():.2f}")
        with c4: st.metric("Range", f"{rng:.2f}")


def create_correlation_heatmap(df, columns):
    if len(columns) < 2:
        st.warning("Need at least 2 variables for correlation heatmap")
        return
    corr_matrix = df[columns].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    # Show strongest correlations
    st.write("**Strongest Correlations (excluding self-correlation):**")
    corr_pairs = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            corr_pairs.append({
                'Variable 1': columns[i],
                'Variable 2': columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs Correlation', ascending=False).drop(columns='Abs Correlation')
    st.dataframe(corr_df.head(10).round(3), hide_index=True)


def create_scatter_matrix(df, columns):
    if len(columns) < 2:
        st.warning("Need at least 2 variables for scatter matrix")
        return
    if len(columns) > 6:
        st.warning("Scatter matrix works best with ≤6 variables. Showing first 6.")
        columns = columns[:6]
    fig = px.scatter_matrix(
        df[columns],
        dimensions=columns,
        title="Scatter Matrix (Pair Plot)"
    )
    fig.update_traces(diagonal_visible=True, showupperhalf=True, marker=dict(size=4, opacity=0.6))
    fig.update_layout(height=150*len(columns), width=150*len(columns))
    st.plotly_chart(fig, width="stretch")
    st.caption("Diagonal shows distribution; off-diagonal shows pairwise scatter plots.")


def create_exploratory_plots(df, plot_types):
    if not plot_types:
        return
    st.header("📊 Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns for visualization")
        return
    viz_cols = st.multiselect("Select variables to visualize:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
    if not viz_cols:
        st.warning("Please select at least one variable to visualize")
        return
    for p in plot_types:
        with st.expander(f"📈 {p} Analysis"):
            if p == "Histogram": create_histograms(df, viz_cols)
            elif p == "Box Plot": create_box_plots(df, viz_cols)
            elif p == "Violin Plot": create_violin_plots(df, viz_cols)
            elif p == "Dot Plot": create_dot_plots(df, viz_cols)
            elif p == "Stem-and-Leaf": create_stem_leaf_plots(df, viz_cols)
            elif p == "Correlation Heatmap": create_correlation_heatmap(df, viz_cols)
            elif p == "Scatter Matrix": create_scatter_matrix(df, viz_cols)


# ──────────────────────────────────────────────────────────────────────────────
# Model Building & Evaluation Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_estimator(model_type, poly_degree, robust_type, is_classifier,
                   reg_alpha=1.0, l1_ratio=0.5,
                   gb_n_estimators=100, gb_learning_rate=0.1, gb_max_depth=3):
    if model_type == "Linear Regression":
        est = LinearRegression()
    elif model_type == "Polynomial Regression":
        est = Pipeline([('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                        ('lin', LinearRegression())])
    elif model_type == "Robust Regression":
        est = HuberRegressor(max_iter=1000) if robust_type == "Huber" else RANSACRegressor(max_trials=1000)
    elif model_type == "Ridge Regression":
        est = Ridge(alpha=reg_alpha)
    elif model_type == "Lasso Regression":
        est = Lasso(alpha=reg_alpha, max_iter=5000)
    elif model_type == "ElasticNet Regression":
        est = ElasticNet(alpha=reg_alpha, l1_ratio=l1_ratio, max_iter=5000)
    elif model_type == "Logistic Regression":
        est = LogisticRegression(max_iter=2000)
    elif model_type == "Random Forest":
        est = RandomForestClassifier(n_estimators=300, random_state=42) if is_classifier else \
              RandomForestRegressor(n_estimators=300, random_state=42)
    elif model_type == "Gradient Boosting":
        est = GradientBoostingClassifier(
            n_estimators=gb_n_estimators, learning_rate=gb_learning_rate,
            max_depth=gb_max_depth, random_state=42
        ) if is_classifier else GradientBoostingRegressor(
            n_estimators=gb_n_estimators, learning_rate=gb_learning_rate,
            max_depth=gb_max_depth, random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return est


def run_hyperparameter_tuning(pipe, model_type, is_classifier, X_train, y_train,
                              method="Random Search", n_iter=20, random_state=42):
    """Run GridSearchCV or RandomizedSearchCV on the full pipeline and return the best pipeline."""
    scoring = "accuracy" if is_classifier else "r2"
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state) if is_classifier \
         else KFold(5, shuffle=True, random_state=random_state)

    # Build param grids keyed to the last step name in the pipeline
    step_name = pipe.steps[-1][0]
    param_grids = {
        "Linear Regression":     {},
        "Ridge Regression":      {f"{step_name}__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
        "Lasso Regression":      {f"{step_name}__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
        "ElasticNet Regression": {f"{step_name}__alpha": [0.001, 0.01, 0.1, 1, 10],
                                  f"{step_name}__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "Polynomial Regression": {f"{step_name}__lin__fit_intercept": [True, False]},
        "Robust Regression":     {},
        "Logistic Regression":   {f"{step_name}__C": [0.01, 0.1, 1, 10, 100],
                                  f"{step_name}__penalty": ["l2"]},
        "Random Forest":         {f"{step_name}__n_estimators": [100, 200, 300],
                                  f"{step_name}__max_depth": [None, 5, 10, 20],
                                  f"{step_name}__min_samples_split": [2, 5, 10]},
        "Gradient Boosting":     {f"{step_name}__n_estimators": [50, 100, 200],
                                  f"{step_name}__learning_rate": [0.01, 0.05, 0.1, 0.2],
                                  f"{step_name}__max_depth": [2, 3, 5]},
    }

    param_grid = param_grids.get(model_type, {})
    if not param_grid:
        # Models with no tunable grid are fitted with the configured settings.
        st.info(f"No hyperparameters to tune for {model_type}. Fitting with the configured settings.")
        pipe.fit(X_train, y_train)
        return pipe, None

    with st.spinner(f"Running {method} ({n_iter} iterations)…"):
        if method == "Grid Search":
            searcher = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        else:
            searcher = RandomizedSearchCV(pipe, param_grid, n_iter=n_iter, cv=cv,
                                          scoring=scoring, n_jobs=-1, refit=True,
                                          random_state=random_state)
        searcher.fit(X_train, y_train)

    st.success(f"✅ Best CV {scoring}: **{searcher.best_score_:.4f}**")
    best_params_clean = {k.replace(f"{step_name}__", ""): v for k, v in searcher.best_params_.items()}
    st.write("**Best parameters:**", best_params_clean)

    results_df = pd.DataFrame(searcher.cv_results_)
    results_df = results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    results_df["params"] = results_df["params"].apply(
        lambda d: {k.replace(f"{step_name}__", ""): v for k, v in d.items()}
    )
    results_df = results_df.sort_values("rank_test_score").head(10)
    st.write("**Top 10 parameter combinations:**")
    st.dataframe(results_df.round(4), hide_index=True)

    return searcher.best_estimator_, searcher.best_params_


def regression_assumptions_block(df, feature_cols, y, preprocessor, X_fit, Xp_fit, feature_names, y_fit=None):
    st.header("🧪 Regression Assumptions & Diagnostics")
    Xnp = Xp_fit.toarray() if hasattr(Xp_fit, 'toarray') else np.asarray(Xp_fit)
    Xsm = sm.add_constant(Xnp, has_constant='add')
    if y_fit is None:
        y_fit = y.loc[X_fit.index].values
    else:
        y_fit = np.asarray(y_fit)
    ols_res = sm.OLS(y_fit, Xsm).fit(cov_type="HC3")

    with st.expander("📦 Multicollinearity (VIF)"):
        max_cols_for_vif = min(Xsm.shape[1]-1, 25)
        names = ['const'] + list(feature_names)
        vif_rows = [{"Feature": names[i], "VIF": variance_inflation_factor(Xsm, i)} for i in range(1, max_cols_for_vif+1)]
        st.dataframe(pd.DataFrame(vif_rows).round(3), hide_index=True)
        st.caption("Rule of thumb: VIF > 5–10 suggests problematic multicollinearity.")

    resid = ols_res.resid
    fitted = ols_res.fittedvalues

    with st.expander("🌡️ Heteroskedasticity (Breusch-Pagan / White)"):
        bp = het_breuschpagan(resid, Xsm)
        wh = het_white(resid, Xsm)
        st.write(f"Breusch-Pagan p = {bp[1]:.4f} | White p = {wh[1]:.4f}")
        st.caption("p < 0.05 → evidence of heteroskedasticity (use robust SEs, transform, or different model).")

    with st.expander("🔄 Autocorrelation (Durbin-Watson)"):
        dw = durbin_watson(resid)
        st.write(f"Durbin-Watson = {dw:.3f}")
        st.caption("≈2 is ideal; <1.5 or >2.5 indicates positive/negative autocorrelation.")

    with st.expander("🧰 Functional Form (Ramsey RESET)"):
        reset = linear_reset(ols_res, power=2, use_f=True)
        st.write(f"RESET F p-value = {reset.pvalue:.4f}")
        st.caption("p < 0.05 → misspecification; consider interactions / nonlinear terms.")

    with st.expander("🧲 Influence (Cook's & Leverage)"):
        infl = OLSInfluence(ols_res)
        cooks = infl.cooks_distance[0]
        lev = infl.hat_matrix_diag
        top_idx = np.argsort(cooks)[-10:][::-1]
        inf_df = pd.DataFrame({"idx": top_idx, "Cook's D": cooks[top_idx], "Leverage": lev[top_idx], "Residual": resid[top_idx]})
        st.dataframe(inf_df.round(4))

    with st.expander("📐 Q–Q Plot (Residual Normality)"):
        r_sorted = np.sort((resid - np.mean(resid)) / (np.std(resid) + 1e-12))
        q_theor = stats.norm.ppf(np.linspace(0.5/len(r_sorted), 1-0.5/len(r_sorted), len(r_sorted)))
        fig = px.scatter(x=q_theor, y=r_sorted, labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"}, title="Q–Q Plot of Residuals")
        fig.add_trace(go.Scatter(x=[q_theor.min(), q_theor.max()], y=[q_theor.min(), q_theor.max()], mode='lines', name='45° line'))
        st.plotly_chart(fig, width="stretch")

    fig2 = px.scatter(x=fitted, y=resid, labels={"x": "Fitted", "y": "Residual"}, title="Residuals vs Fitted")
    fig2.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig2, width="stretch")


def classification_block(y_test, y_pred_test, y_proba_test):
    st.header("🧪 Classification Diagnostics")
    classes = sorted(pd.Series(y_test).unique())
    n_classes = len(classes)
    is_binary = n_classes == 2

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, normalize='true')
    fig_cm = px.imshow(
        cm, text_auto=".2f", aspect='auto',
        title="Confusion Matrix (True-class normalized)",
        x=[str(c) for c in classes], y=[str(c) for c in classes],
        labels={"x": "Predicted", "y": "Actual"},
    )
    st.plotly_chart(fig_cm, width="stretch")

    # Classification report with macro/weighted summary
    st.text("Classification report:\n" + classification_report(y_test, y_pred_test, digits=3))
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_test):.4f}")
    with c2: st.metric("F1 (macro)", f"{f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")
    with c3: st.metric("F1 (weighted)", f"{f1_score(y_test, y_pred_test, average='weighted', zero_division=0):.4f}")

    if y_proba_test is None:
        st.info("This model does not expose predicted probabilities, so ROC, precision-recall and Brier outputs are unavailable.")
        return

    y_bin = None if is_binary else label_binarize(y_test, classes=classes)

    # For a binary target, name the positive class explicitly. Column 1 of
    # predict_proba corresponds to classes[1] because sklearn orders classes_ by
    # sort order, which is how `classes` is built above. Without pos_label these
    # calls raise for any binary target that is not labelled {0, 1}.
    pos_label = classes[1] if is_binary else None
    pos_score = None
    if is_binary:
        pos_score = y_proba_test[:, 1] if getattr(y_proba_test, "ndim", 1) == 2 else y_proba_test

    # ROC curves
    with st.expander("📈 ROC Curves"):
        if is_binary:
            fpr, tpr, _ = roc_curve(y_test, pos_score, pos_label=pos_label)
            roc_auc_val = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy', name=f"AUC={roc_auc_val:.3f}"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name="Random"))
            fig_roc.update_layout(title=f"ROC Curve (positive class = {pos_label})",
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, width="stretch")
        else:
            # One-vs-Rest ROC per class
            fig_roc = go.Figure()
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_test[:, i])
                roc_auc_val = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Class {cls} (AUC={roc_auc_val:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name="Random"))
            fig_roc.update_layout(title="ROC Curves (One-vs-Rest)", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, width="stretch")
            macro_auc = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro')
            st.metric("ROC-AUC (macro OvR)", f"{macro_auc:.4f}")

    # Precision-Recall curves
    with st.expander("📉 Precision-Recall Curves"):
        if is_binary:
            prec, rec, _ = precision_recall_curve(y_test, pos_score, pos_label=pos_label)
            pr_auc_val = auc(rec, prec)
            base_rate = float(np.mean(np.asarray(pd.Series(y_test).values) == pos_label))
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, fill='tozeroy', name=f"AUC={pr_auc_val:.3f}"))
            fig_pr.add_hline(y=base_rate, line_dash="dash",
                             annotation_text=f"No-skill baseline ({base_rate:.3f})")
            fig_pr.update_layout(title=f"Precision-Recall Curve (positive class = {pos_label})",
                                 xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, width="stretch")
            # brier_score_loss needs 0/1 targets (or an explicit positive label);
            # map to 0/1 so the metric works for any binary labelling.
            y_true_01 = (np.asarray(pd.Series(y_test).values) == pos_label).astype(int)
            bs = brier_score_loss(y_true_01, pos_score)
            st.metric("Brier score (↓ better)", f"{bs:.4f}")
            st.caption("Brier score is the mean squared error of the predicted probabilities: 0 is perfect, "
                       f"and predicting the base rate ({base_rate:.3f}) for every case would score "
                       f"{base_rate * (1 - base_rate):.4f}.")
        else:
            fig_pr = go.Figure()
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba_test[:, i])
                pr_auc_val = auc(rec, prec)
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, name=f"Class {cls} (AUC={pr_auc_val:.3f})"))
            fig_pr.update_layout(title="Precision-Recall Curves (One-vs-Rest)", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, width="stretch")


def run_data_diagnostics(df):
    st.subheader("🔍 Data Quality Diagnostics")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        missing_count = int(df.isnull().sum().sum())
        st.metric("Missing Values", missing_count)
        if missing_count == 0:
            st.success("✅ No missing values")
        else:
            st.warning("⚠️ Contains missing values")

    with c2:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.metric("Infinite Values", "N/A")
            st.info("No numeric columns")
        else:
            inf_count = int(np.isinf(numeric_df.values).sum())
            st.metric("Infinite Values", inf_count)
            if inf_count == 0:
                st.success("✅ No infinite values")
            else:
                st.error("❌ Contains infinite values")

    with c3:
        n_samples = int(len(df))
        st.metric("Sample Size", n_samples)
        if n_samples >= 30:
            st.success("✅ Adequate sample size")
        else:
            st.warning("⚠️ Small sample size")

    with c4:
        duplicates = int(df.duplicated().sum())
        st.metric("Duplicate Rows", duplicates)
        if duplicates == 0:
            st.success("✅ No duplicates")
        else:
            st.warning("⚠️ Contains duplicates")


# ──────────────────────────────────────────────────────────────────────────────
# Overall-F support
#
# The overall-F statistic F = (R²/k) / ((1-R²)/(n-k-1)) has an F(k, n-k-1) null
# distribution ONLY for an ordinary least-squares fit. It is not valid for
# Ridge/Lasso/ElasticNet (shrinkage changes the effective degrees of freedom),
# Huber/RANSAC (not least squares at all), the tree ensembles (no fixed parameter
# count), or any classifier, where R² on class labels means nothing. The test is
# computed only where it is valid, and the app says why everywhere else.
# ──────────────────────────────────────────────────────────────────────────────

F_TEST_INAPPLICABLE = {
    "Logistic Regression":
        "There is no residual sum of squares to partition. Judge the fit from the "
        "classification diagnostics and the cross-validation scores instead.",
    "Ridge Regression":
        "Ridge shrinks the coefficients toward zero, so the effective degrees of freedom "
        "are not the column count and the F distribution does not hold.",
    "Lasso Regression":
        "Lasso shrinks and selects, so the number of parameters is data-dependent and the "
        "F distribution does not hold.",
    "ElasticNet Regression":
        "ElasticNet shrinks and selects, so the number of parameters is data-dependent and "
        "the F distribution does not hold.",
    "Robust Regression":
        "Huber and RANSAC do not minimise the residual sum of squares, so the fit has no F "
        "reference distribution.",
    "Random Forest":
        "An ensemble has no fixed parameter count, so there are no degrees of freedom to "
        "test against.",
    "Gradient Boosting":
        "An ensemble has no fixed parameter count, so there are no degrees of freedom to "
        "test against.",
}

F_TEST_DEFAULT_REASON = (
    "The overall-F statistic assumes an ordinary least-squares fit."
)


def ols_design_matrix(model, X):
    """Design matrix the fitted estimator actually used, if and only if it is OLS.

    Linear Regression is OLS on the preprocessed matrix as supplied. Polynomial
    Regression is OLS on that matrix expanded by PolynomialFeatures, so the expansion
    is applied here — using X unexpanded would understate k and inflate the F statistic.
    Every other model family returns None.
    """
    try:
        Xv = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        return None

    if isinstance(model, LinearRegression):
        return Xv

    if isinstance(model, Pipeline):
        steps = model.named_steps
        if "poly" in steps and isinstance(steps.get("lin"), LinearRegression):
            try:
                return np.asarray(steps["poly"].transform(Xv), dtype=float)
            except Exception:
                return None

    return None


def ols_effective_rank(design):
    """Number of linearly independent non-constant regressors in `design`.

    Uses the rank rather than the column count so that a collinear design — a full
    one-hot dummy set, or a polynomial expansion of one, where an indicator squared is
    itself — does not overstate the numerator degrees of freedom. Falls back to the
    column count if the rank cannot be computed.
    """
    n_cols = int(design.shape[1])
    try:
        with_const = np.column_stack([np.ones(len(design)), design])
        return int(np.linalg.matrix_rank(with_const)) - 1, n_cols
    except (np.linalg.LinAlgError, ValueError, MemoryError):
        return n_cols, n_cols


def reference_levels(pipe, cat_cols):
    """{categorical column: its reference (dropped) level} from the fitted encoder."""
    if not cat_cols:
        return {}
    try:
        enc = pipe[0].named_transformers_["cat"]
        out = {}
        for col, cats, drop in zip(cat_cols, enc.categories_, enc.drop_idx_):
            if drop is not None:
                out[col] = cats[int(drop)]
        return out
    except Exception:
        return {}


def leakage_check(df, target_col, feature_cols, is_classifier):
    """Warn when a feature looks like it was derived from the target.

    Two signatures, both of which make a model look far better than it is:
    - binary target: a numeric feature whose value ranges for the two classes do not
      overlap, so a single threshold on it predicts the target perfectly;
    - numeric target (regression): a numeric feature with |r| > 0.99 with the target.
    A warning only, because a genuinely near-deterministic input is possible.
    """
    y = df[target_col]
    flagged = []
    for c in feature_cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = df[c]
        if is_classifier and y.nunique() == 2:
            a, b = (x[y == v].dropna() for v in sorted(y.dropna().unique()))
            if len(a) and len(b) and (a.max() < b.min() or b.max() < a.min()):
                flagged.append(f"`{c}` perfectly separates the two classes of `{target_col}`")
        elif not is_classifier and pd.api.types.is_numeric_dtype(y):
            r = x.corr(y)
            if pd.notna(r) and abs(r) > 0.99:
                flagged.append(f"`{c}` has |r| = {abs(r):.3f} with `{target_col}`")
    if flagged:
        st.warning(
            "⚠️ **Possible target leakage:** " + "; ".join(flagged) + ". A feature like this is "
            "usually computed from the target, and it will make the model look near-perfect. "
            "Remove it from the features unless it really is an independent input."
        )


def run_statistical_tests(X, y, y_pred, model, model_type, tests_selected, poly_degree, robust_type, cv_pipeline=None, X_raw=None, y_raw=None, ref_levels=None):
    if not tests_selected:
        return
    st.header("📊 Statistical Tests")
    for test in tests_selected:
        with st.expander(f"🔬 {test}"):
            if test == "F-test (Overall Significance)":
                design = ols_design_matrix(model, X)

                if design is not None:
                    n = len(y)
                    r2 = r2_score(y, y_pred)
                    k, n_cols = ols_effective_rank(design)
                    dof = n - k - 1
                    if k > 0 and dof > 0 and 0 < r2 < 1:
                        f_stat = (r2 / k) / ((1 - r2) / dof)
                        f_p = 1 - stats.f.cdf(f_stat, k, dof)
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("F-statistic", f"{f_stat:.3f}")
                        with c2: st.metric("P-value", f"{f_p:.4f}")
                        with c3: st.metric("df (num, den)", f"{k}, {dof}")
                        if f_p < 0.05:
                            st.success("✅ Significant (p<0.05) — the model explains more variance than an intercept-only model")
                        else:
                            st.warning("⚠️ Not significant — no evidence the model beats predicting the mean")
                        if k < n_cols:
                            st.caption(
                                f"Numerator df is the rank of the fitted design ({k}), below its column "
                                f"count ({n_cols}) because some columns are linearly dependent — usually a "
                                "polynomial expansion of indicator columns, where an indicator squared equals "
                                "itself. Using the column count here would overstate F."
                            )
                        st.caption(
                            "Computed on the training set, so R² here is higher than the test R² reported "
                            "above. The test asks only whether the model beats an intercept; it says nothing "
                            "about which predictors matter, or whether the assumptions hold."
                        )
                    else:
                        st.info(
                            f"F-test not computable: R²={r2:.4f}, numerator df={k}, residual df={dof}. "
                            "This needs 0 < R² < 1, at least one independent regressor, and at least one "
                            "residual degree of freedom."
                        )
                else:
                    reason = F_TEST_INAPPLICABLE.get(model_type, F_TEST_DEFAULT_REASON)
                    st.info(f"**F-test not applicable to {model_type}.** {reason}")
                    if not sk_is_classifier(model):
                        st.metric("Training R² (descriptive only)", f"{r2_score(y, y_pred):.4f}")
                        st.caption(
                            "Shown for reference, not as a test. Use the holdout metrics and the "
                            "cross-validation spread to judge this model."
                        )
                    else:
                        st.caption(
                            "For a classification task, judge the fit from accuracy, ROC-AUC and the "
                            "classification diagnostics rather than an R²-based statistic."
                        )

            elif test == "T-test (Individual Coefficients)":
                if model_type == "Linear Regression":
                    try:
                        n, k = len(y), X.shape[1]
                        dof = n - k - 1
                        if dof <= 0:
                            st.warning(
                                f"⚠️ Not enough observations for coefficient t-tests: n={n}, "
                                f"design columns={k}, residual df={dof}. Add data or reduce features."
                            )
                        else:
                            # Unbiased error variance SSE/(n-k-1). SSE/n would make the
                            # standard errors too small and the p-values anti-conservative.
                            resid_tt = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
                            sigma2 = float(np.sum(resid_tt ** 2) / dof)
                            X_i = np.column_stack([np.ones(len(X)), X])
                            XtX_inv = np.linalg.inv(X_i.T @ X_i)
                            se = np.sqrt(np.diag(XtX_inv) * sigma2)
                            coefs = np.concatenate([[model.intercept_], model.coef_])
                            names = ['Intercept'] + list(X.columns)
                            t_stats = coefs / se
                            p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
                            coef_df = pd.DataFrame({
                                'Variable': names,
                                'Coefficient': coefs,
                                'Std. Error': se,
                                'T-statistic': t_stats,
                                'P-value': p_vals,
                                'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in p_vals]
                            })
                            st.dataframe(coef_df, hide_index=True)
                            st.caption(
                                f"Residual df = {dof}. Coefficients are on the preprocessed scale "
                                "(per standard deviation for numeric features; for a categorical, each "
                                "level's difference from the reference level — the first in sorted order, "
                                "which has no row of its own). Standard errors are classical, not heteroskedasticity-robust — "
                                "check the Breusch-Pagan / White results before relying on these p-values."
                            )
                            if ref_levels:
                                refs = ", ".join(f"`{c}` = **{lvl}**" for c, lvl in ref_levels.items())
                                st.caption(f"📌 Reference levels (absorbed into the intercept): {refs}. "
                                           "Each categorical row above is that level minus its reference.")
                    except np.linalg.LinAlgError:
                        st.error("Could not compute t-tests (multicollinearity)")
                else:
                    st.info(f"T-tests not applicable for {model_type}")

            elif test in ["Normality Tests (Shapiro-Wilk)", "Normality Tests (Jarque-Bera)"]:
                if model_type != "Logistic Regression":
                    resid = y - y_pred
                    if test == "Normality Tests (Shapiro-Wilk)":
                        if len(resid) <= 5000:
                            _, p = shapiro(resid)
                            st.metric("Shapiro-Wilk P-value", f"{p:.4f}")
                            if p > 0.05:
                                st.success("✅ Residuals appear normal")
                            else:
                                st.warning("⚠️ Non-normal residuals")
                        else:
                            st.info("Dataset too large for Shapiro-Wilk")
                    else:
                        _, p = jarque_bera(resid)
                        st.metric("Jarque-Bera P-value", f"{p:.4f}")
                        if p > 0.05:
                            st.success("✅ Residuals appear normal")
                        else:
                            st.warning("⚠️ Non-normal residuals")
                else:
                    st.info("Normality tests not applicable for classification")

            elif test == "Cross Validation":
                try:
                    is_classifier = isinstance(
                        model, (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression)
                    )
                    scoring = 'accuracy' if is_classifier else 'r2'
                    if cv_pipeline is not None:
                        cv_est = clone(cv_pipeline)
                    else:
                        cv_est = model
                    cv_X = X_raw if (cv_pipeline is not None and X_raw is not None) else X
                    cv_y = y_raw if (cv_pipeline is not None and y_raw is not None) else y
                    cv_scores = cross_val_score(cv_est, cv_X, cv_y, cv=5, scoring=scoring)
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("CV Mean", f"{cv_scores.mean():.4f}")
                    with c2: st.metric("CV Std", f"{cv_scores.std():.4f}")
                    with c3: st.metric("CV Range", f"{(cv_scores.max() - cv_scores.min()):.4f}")
                    cv_df = pd.DataFrame({'Fold': range(1, 6), 'Score': cv_scores})
                    fig_cv = px.bar(cv_df, x='Fold', y='Score', title=f"Cross-Validation {scoring.title()} Scores")
                    fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red")
                    st.plotly_chart(fig_cv, width="stretch")
                except Exception as e:
                    st.error(f"Cross-validation failed: {str(e)}")

            elif test == "Residual Analysis":
                if model_type != "Logistic Regression":
                    resid = y - y_pred
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Residual Statistics**")
                        mean_resid = float(np.mean(resid))
                        st.write(f"Mean: {mean_resid:.6f}")
                        st.write(f"Std Dev: {np.std(resid):.4f}")
                        if abs(mean_resid) < 1e-10:
                            st.success("✅ Mean residual ≈ 0")
                        else:
                            st.warning(f"⚠️ Mean residual = {mean_resid:.6f}")
                    with c2:
                        fig_hist = px.histogram(x=resid, nbins=20, title="Residual Distribution")
                        st.plotly_chart(fig_hist, width="stretch")
                    fig_resid = px.scatter(x=y_pred, y=resid, title="Residuals vs Fitted Values")
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_resid.update_layout(xaxis_title="Fitted Values", yaxis_title="Residuals")
                    st.plotly_chart(fig_resid, width="stretch")
                else:
                    st.info("Residual analysis not applicable for classification")


def run_model_validation_tests(model, X, y, y_pred, is_classifier):
    st.subheader("🧪 Model Validation Tests")
    rows = []
    if is_classifier:
        acc = accuracy_score(y, y_pred)
        rows.append({'Test': 'Accuracy bounds (0-1)', 'Result': '✅ Pass' if 0 <= acc <= 1 else '❌ Fail', 'Value': f"{acc:.4f}"})
        rows.append({'Test': 'Predictions are valid classes',
                     'Result': '✅ Pass' if set(pd.Series(y_pred).unique()).issubset(set(pd.Series(y).unique())) else '❌ Fail',
                     'Value': f"{len(set(y_pred))} classes"})
    else:
        r2 = r2_score(y, y_pred)
        rows.append({'Test': 'R² bounds check', 'Result': '✅ Pass' if r2 <= 1 else '❌ Fail', 'Value': f"{r2:.4f}"})
        rows.append({'Test': 'No NaN predictions',
                     'Result': '✅ Pass' if not np.any(np.isnan(y_pred)) else '❌ Fail',
                     'Value': f"{np.sum(np.isnan(y_pred))} NaN"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    passed = sum(1 for r in rows if '✅' in r['Result'])
    total = len(rows)
    if passed == total:
        st.success(f"🎉 All {total} validation tests passed!")
    else:
        st.warning(f"⚠️ {passed}/{total} validation tests passed")


# ──────────────────────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────────────────────

def show_quick_start():
    """Landing content, shown only until data is loaded."""
    st.markdown(
        "PostRun takes you from raw simulation output to diagnosed statistical models, "
        "with no code. Expect **one row per run or replication**: the parameters you varied "
        "as columns, and the outcomes you measured as columns."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1 · Load data**\n\nUpload a CSV or Excel file in the sidebar, or click "
                    "**Load example: supply-network simulation** to try it out.")
    with c2:
        st.markdown("**2 · Pick target and features**\n\nChoose the outcome to explain and the "
                    "inputs to explain it with. Explore the plots first if you like.")
    with c3:
        st.markdown("**3 · Choose a model**\n\nPick one in the sidebar. Results, diagnostics and "
                    "statistical tests appear below and update as you change settings.")
    st.info("**First run?** Load the example, keep `fill_rate` as the target, and choose "
            "*Linear Regression*. Then switch the target to `met_target` and choose "
            "*Logistic Regression* (remove `fill_rate` from the features first).")
    st.caption("🔒 Uploaded files are processed on this app's host. Don't upload sensitive or "
               "proprietary data; run PostRun locally for that.")


@st.cache_data
def load_example_dataset() -> pd.DataFrame:
    return pd.read_csv(EXAMPLE_PATH)


def main():
    # Load data. An uploaded file takes priority; otherwise use the sample dataset the
    # user chose, which stays loaded across reruns until they clear it.
    source = st.session_state.get("data_source")
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            sheet_names = pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
            sheet = st.selectbox("Select sheet to load:", sheet_names) if len(sheet_names) > 1 else sheet_names[0]
        else:
            sheet = None
        df = load_uploaded_data(file_bytes, uploaded_file.name, sheet)
        process = validate_data(df)
        if source:
            st.caption("Using your uploaded file. Remove it in the sidebar to go back to the sample data.")
    elif source == "example":
        if not EXAMPLE_PATH.exists():
            st.error("The example dataset (sample_data.csv) is missing from the app folder.")
            return
        df = load_example_dataset()
        st.info("📝 **Example dataset:** 500 synthetic runs of a supply-network simulation. "
                "`fill_rate` is the continuous outcome; `met_target` (fill_rate ≥ 0.5) is the "
                "binary outcome; `policy` is a 3-level categorical; `replication` is a run ID.")
        st.download_button("📥 Download example CSV", df.to_csv(index=False), "sample_data.csv", "text/csv")
        process = True
    elif source == "generated":
        df = generate_sample_dataset()
        st.info("📝 Using generated sample dataset")
        st.download_button("📥 Download Sample Dataset", df.to_csv(index=False), "generated_data.csv", "text/csv")
        process = True
    else:
        show_quick_start()
        process = False
        df = None

    if not process:
        return

    # 1. Dataset Overview
    st.header("📋 Dataset Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
    with c2:
        dtype_df = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str), 'Non-Null': df.count()})
        st.dataframe(dtype_df, hide_index=True)

    # 2. Missing Value Imputation
    df = handle_missing_values(df)

    # 3. Outlier Treatment
    df = handle_outliers(df)

    # 4. Data Quality Diagnostics
    if show_diagnostic_tests:
        run_data_diagnostics(df)

    # 5. EDA (before variable selection)
    if plot_types:
        create_exploratory_plots(df, plot_types)

    # 6. Variable Selection (after EDA)
    X, y, target_col, feature_cols = select_variables(df)
    if X is None:
        return

    st.success(f"✅ Selected target: **{target_col}** | Features: **{len(feature_cols)}**")

    # 7. Feature Engineering
    df, feature_cols = apply_feature_engineering(df, feature_cols)

    # 8. Determine task type before split so stratify is correct
    regression_models = {
        "Linear Regression", "Polynomial Regression", "Robust Regression",
        "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
    }
    classification_only = {"Logistic Regression"}
    if model_type in regression_models or task_type == "Regression":
        is_classifier = False
    elif model_type in classification_only or task_type == "Classification":
        is_classifier = True
    else:  # Auto-detect for Random Forest / Gradient Boosting
        is_classifier = (
            y.dtype in ['object', 'category', 'bool'] or
            (pd.api.types.is_integer_dtype(y) and y.nunique() <= 10)
        )

    # 9. Preprocessing & Train/Test Split
    # Guard: a classification task needs a target with a small number of classes, each
    # with at least 2 rows, or the stratified split (and every CV fold after it) fails.
    # This catches the common slip of choosing a classifier while a continuous or ID
    # column is still selected as the target.
    if is_classifier:
        class_counts = pd.Series(y).value_counts()
        if y.nunique() > 20 or class_counts.min() < 2:
            st.error(
                f"**`{target_col}` doesn't look like a classification target** — it has "
                f"{y.nunique()} distinct values, and the smallest class has "
                f"{int(class_counts.min())} row(s). Choose a categorical or binary target "
                "(each class needs at least 2 rows), or pick a regression model."
            )
            return

    leakage_check(df, target_col, feature_cols, is_classifier)

    preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_cols)
    stratify = y if is_classifier else None
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Fit preprocessor early to get feature names for the sample-size check
    preprocessor.fit(X_train)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"x{i}" for i in range(len(feature_cols))]

    # 10. Sample Size Adequacy Check
    st.subheader("🔢 Sample Size Adequacy (model-aware)")
    if model_type == "Logistic Regression" and pd.Series(y).nunique() != 2:
        n = int(len(y))
        st.metric("Sample Size", n)
        st.warning("Target is not binary; logistic EPV check not applicable.")
    else:
        ok, n, msg = sample_size_advice(model_type, y, p=len(feature_names))
        st.metric("Sample Size", n)
        if ok:
            st.success(f"✅ Adequate sample size — {msg}")
        else:
            st.warning(f"⚠️ Possibly underpowered — {msg}")

    # 11. Build & Fit Pipeline (preprocessor is reused; pipeline refits it on same data)
    est = make_estimator(
        model_type, poly_degree, robust_type, is_classifier,
        reg_alpha=reg_alpha, l1_ratio=l1_ratio,
        gb_n_estimators=gb_n_estimators, gb_learning_rate=gb_learning_rate, gb_max_depth=gb_max_depth,
    )
    pipe = make_pipeline(preprocessor, est)

    if enable_tuning:
        st.subheader("🔧 Hyperparameter Tuning")
        pipe, _ = run_hyperparameter_tuning(
            pipe, model_type, is_classifier, X_train, y_train,
            method=tuning_method, n_iter=tuning_n_iter, random_state=random_state,
        )
    else:
        pipe.fit(X_train, y_train)

    # Extract preprocessed arrays from the fitted pipeline (single source of truth)
    Xp_train = pipe[:-1].transform(X_train)
    Xp_test = pipe[:-1].transform(X_test)
    try:
        feature_names = pipe[:-1].get_feature_names_out()
    except Exception:
        feature_names = [f"x{i}" for i in range(Xp_train.shape[1])]

    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    y_proba_test = None
    if is_classifier and hasattr(pipe[-1], "predict_proba"):
        try:
            y_proba_test = pipe.predict_proba(X_test)
        except Exception:
            y_proba_test = None

    # Model Performance
    st.header(f"🤖 {model_type} Analysis")
    st.subheader("📏 Holdout Performance (Test set)")
    if is_classifier:
        acc = accuracy_score(y_test, y_pred_test)
        n_classes = pd.Series(y_test).nunique()
        st.metric("Accuracy (test)", f"{acc:.4f}")
        if y_proba_test is not None:
            try:
                if n_classes == 2:
                    auc_roc = roc_auc_score(y_test, y_proba_test[:, 1])
                else:
                    auc_roc = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro')
                st.metric("ROC-AUC (test)", f"{auc_roc:.4f}")
            except Exception:
                pass
    else:
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = np.mean(np.abs(y_test - y_pred_test))
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("R² (test)", f"{r2:.4f}")
        with c2: st.metric("RMSE (test)", f"{rmse:.4f}")
        with c3: st.metric("MAE (test)", f"{mae:.4f}")

    # Performance Visualization
    st.subheader("📈 Model Performance Visualization")
    if is_classifier:
        cm = confusion_matrix(y_test, y_pred_test)
        fig_cm = px.imshow(cm, text_auto=True, aspect='auto', title="Confusion Matrix (counts)")
        st.plotly_chart(fig_cm, width="stretch")
        classification_block(y_test, y_pred_test, y_proba_test)
    else:
        fig_pred = px.scatter(x=y_test, y=y_pred_test, title="Actual vs Predicted (Test)")
        mn, mx = min(pd.Series(y_test).min(), pd.Series(y_pred_test).min()), max(pd.Series(y_test).max(), pd.Series(y_pred_test).max())
        fig_pred.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Perfect', line=dict(color='red', dash='dash')))
        fig_pred.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig_pred, width="stretch")

    # Learning Curve
    with st.expander("📉 Learning Curve"):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if is_classifier else KFold(n_splits=5, shuffle=True, random_state=random_state)
        sizes, tr_scores, te_scores = learning_curve(pipe, df[feature_cols], y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5))
        lc_df = pd.DataFrame({"Train size": np.r_[sizes, sizes], "Score": np.r_[tr_scores.mean(axis=1), te_scores.mean(axis=1)], "Split": ["Train"]*len(sizes)+["CV"]*len(sizes)})
        fig_lc = px.line(lc_df, x="Train size", y="Score", color="Split", markers=True, title="Learning Curve")
        st.plotly_chart(fig_lc, width="stretch")

    # Model Validation Tests
    if show_validation_tests:
        run_model_validation_tests(pipe, X_test, y_test, y_pred_test, is_classifier)

    # Statistical Tests
    X_dense = Xp_train.toarray() if hasattr(Xp_train, "toarray") else np.asarray(Xp_train)
    run_statistical_tests(
        X=pd.DataFrame(X_dense, columns=feature_names),
        y=y_train.reset_index(drop=True),
        y_pred=pd.Series(pipe.predict(X_train)).reset_index(drop=True),
        model=pipe[-1],
        model_type=model_type,
        tests_selected=statistical_tests,
        poly_degree=poly_degree,
        robust_type=robust_type,
        cv_pipeline=pipe,
        X_raw=df.loc[y_train.index, feature_cols],
        y_raw=y_train,
        ref_levels=reference_levels(pipe, cat_cols),
    )

    # Feature Importance
    if show_feature_analysis and hasattr(pipe[-1], 'feature_importances_'):
        st.subheader("🎯 Feature Importance (Model)")
        importances = pipe[-1].feature_importances_
        n = min(len(importances), len(feature_names))
        imp_df = pd.DataFrame({'Feature': list(feature_names)[:n], 'Importance': importances[:n]})
        imp_df = imp_df.sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig_imp, width="stretch")

    # Regression Assumptions & Diagnostics (use training data for reliable OLS diagnostics)
    if not is_classifier:
        regression_assumptions_block(df, feature_cols, y, preprocessor, X_train, Xp_train, feature_names, y_fit=y_train)

    # ANOVA
    cat_candidates = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in cat_candidates:
        cat_candidates.remove(target_col)
    anova_factor = None
    if cat_candidates:
        anova_factor = st.sidebar.selectbox("One-way ANOVA factor (categorical)", ["(none)"] + cat_candidates)
    if anova_factor and anova_factor != "(none)" and pd.api.types.is_numeric_dtype(df[target_col]):
        with st.expander("📊 One-way ANOVA + Tukey HSD"):
            tmp = df[[anova_factor, target_col]].dropna().copy()
            model_aov = smf_ols(f"{target_col} ~ C({anova_factor})", data=tmp).fit()
            aov_table = sm.stats.anova_lm(model_aov, typ=2)
            st.dataframe(aov_table.round(4))
            tuk = pairwise_tukeyhsd(endog=tmp[target_col].values, groups=tmp[anova_factor].values, alpha=0.05)
            st.text(str(tuk))
            ss_between = aov_table.loc[f"C({anova_factor})", "sum_sq"]
            ss_total = ss_between + aov_table.loc["Residual", "sum_sq"]
            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
            st.metric("η² (effect size)", f"{eta_sq:.3f}")


if __name__ == "__main__":
    main()
