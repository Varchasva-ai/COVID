"""
COVID-19 Early Case Trend Analysis & Recovery Insights - WEB APP VERSION
HealthGuard Analytics Pvt. Ltd. - Minor Project

Calculates each patient's AGE from their birth_year (age = reference_year -
birth_year). If your dataset already has a direct 'age' column instead,
the script uses that automatically.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

This turns the analysis into an interactive website where you (or anyone)
can upload a CSV and instantly see all the charts, stats, and regression
results in the browser.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Case Analysis | HealthGuard Analytics",
    page_icon="🦠",
    layout="wide",
)
sns.set_style("whitegrid")

st.title("🦠 COVID-19 Early Case Trend Analysis & Recovery Insights")
st.caption("HealthGuard Analytics Pvt. Ltd. — Infectious Disease Case Analysis Dashboard")

# ----------------------------------------------------------------------
# SIDEBAR - FILE UPLOAD & SETTINGS
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload patient dataset (CSV)", type="csv")
reference_year = st.sidebar.number_input(
    "Reference year (used to calculate age = reference year - birth year)",
    value=2020,
    step=1,
)

if uploaded_file is None:
    st.info("👈 Upload your dataset CSV from the sidebar to generate the dashboard.")
    st.stop()

# ----------------------------------------------------------------------
# LOAD & AUTO-MAP COLUMNS
# ----------------------------------------------------------------------
df = pd.read_csv(uploaded_file)

ALIASES = {
    "sex": ["sex", "gender"],
    "age": ["age"],
    "birth_year": ["birth_year", "birthyear"],
    "country": ["country"],
    "region": ["region", "province", "state_region"],
    "infection_reason": ["infection_reason", "infection_case", "reason"],
    "infection_order": ["infection_order"],
    "infected_by": ["infected_by"],
    "contact_number": ["contact_number", "contact_no", "contacts"],
    "confirmed_date": ["confirmed_date", "date_confirmation"],
    "released_date": ["released_date", "date_released"],
    "deceased_date": ["deceased_date", "date_death"],
    "state": ["state", "case_outcome", "status"],
}

lower_cols = {c.lower().strip(): c for c in df.columns}
rename_map, found_log = {}, {}
for standard_name, candidates in ALIASES.items():
    for candidate in candidates:
        if candidate in lower_cols:
            rename_map[lower_cols[candidate]] = standard_name
            found_log[standard_name] = lower_cols[candidate]
            break
df = df.rename(columns=rename_map)
for standard_name in ALIASES:
    if standard_name not in df.columns:
        df[standard_name] = np.nan

with st.sidebar.expander("📋 Column mapping detected"):
    for standard_name in ALIASES:
        if standard_name in found_log:
            st.write(f"✅ **{standard_name}** ← `{found_log[standard_name]}`")
        else:
            st.write(f"⚠️ **{standard_name}** — not found")

# ----------------------------------------------------------------------
# CLEAN DATA
# ----------------------------------------------------------------------
for col in ["confirmed_date", "released_date", "deceased_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# ---- AGE CALCULATION ----
# Priority 1: derive age from birth_year (age = reference_year - birth_year)
# Priority 2: use a direct 'age' column if birth_year isn't available
#             (handles both numeric ages like 35 and bucketed strings like "30s")
if df["birth_year"].notna().sum() > 0:
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age"] = reference_year - df["birth_year"]
    st.sidebar.success(f"Age calculated as {reference_year} − birth_year ✅")
elif df["age"].notna().sum() > 0:
    def parse_age(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s.endswith("s"):
            s = s[:-1]
        try:
            return float(s)
        except ValueError:
            return np.nan
    df["age"] = df["age"].apply(parse_age)
    st.sidebar.info("Using the dataset's existing 'age' column (no birth_year found).")
else:
    st.sidebar.warning("No birth_year or age column found — age-based charts will be skipped.")

# Drop impossible ages (data entry errors)
df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan

for col in ["sex", "country", "region", "infection_reason", "state"]:
    df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", np.nan)

for col in ["contact_number", "infection_order"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

released = df[df["released_date"].notna() & df["confirmed_date"].notna()].copy()
released["recovery_duration"] = (released["released_date"] - released["confirmed_date"]).dt.days
released = released[released["recovery_duration"] >= 0]

# ----------------------------------------------------------------------
# TOP-LEVEL METRICS
# ----------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Records", f"{len(df):,}")
m2.metric("Released Cases", f"{(df['state'] == 'released').sum():,}")
m3.metric("Deceased Cases", f"{(df['state'] == 'deceased').sum():,}")
m4.metric("Avg. Recovery Days", f"{released['recovery_duration'].mean():.1f}" if len(released) else "N/A")

st.divider()

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["👥 Demographics", "🔄 Spread Patterns", "📈 Recovery Trends", "🗺️ Regional Impact", "📊 Regression Model"]
)

# ---- TAB 1: DEMOGRAPHICS ----
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        if df["sex"].notna().sum() > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(data=df, x="sex", order=df["sex"].value_counts().index, ax=ax)
            ax.set_title("Confirmed Cases by Gender")
            st.pyplot(fig)
        else:
            st.warning("No gender data found.")
    with col2:
        if df["age"].notna().sum() > 0:
            bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 120]
            labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
            df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="age_group", order=labels, ax=ax)
            ax.set_title("Confirmed Cases by Age Group")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No age data found.")

    if df["age"].notna().sum() > 0:
        st.subheader("Age Summary")
        st.dataframe(df["age"].describe().to_frame(name="Age (years)"))

    if df["region"].notna().sum() > 0:
        top_regions = df["region"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=top_regions.values, y=top_regions.index, ax=ax)
        ax.set_title("Top 10 Regions by Confirmed Cases")
        st.pyplot(fig)
    else:
        st.warning("No region data found.")

# ---- TAB 2: SPREAD PATTERNS ----
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        if df["infection_reason"].notna().sum() > 0:
            top_reasons = df["infection_reason"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(x=top_reasons.values, y=top_reasons.index, ax=ax)
            ax.set_title("Top Infection Reasons")
            st.pyplot(fig)
        else:
            st.warning("No infection reason data found.")
    with col2:
        if df["contact_number"].notna().sum() > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.histplot(df["contact_number"].dropna(), bins=30, ax=ax)
            ax.set_title("Distribution of Contact Numbers")
            st.pyplot(fig)
        else:
            st.warning("No contact number data found.")

    if df["infection_order"].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df["infection_order"].dropna(), bins=20, ax=ax)
        ax.set_title("Distribution of Infection Order")
        st.pyplot(fig)

# ---- TAB 3: RECOVERY TRENDS ----
with tab3:
    if len(released) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(released["recovery_duration"], bins=30, kde=True, ax=ax)
            ax.set_title("Recovery Duration Distribution")
            st.pyplot(fig)
        with col2:
            weekly_avg = released.set_index("confirmed_date")["recovery_duration"].resample("W").mean()
            fig, ax = plt.subplots(figsize=(6, 4))
            weekly_avg.plot(marker="o", ax=ax)
            ax.set_title("Avg Recovery Duration Over Time")
            st.pyplot(fig)

        if released["region"].notna().sum() > 0:
            top10 = released["region"].value_counts().head(10).index
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=released[released["region"].isin(top10)], x="region", y="recovery_duration", ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Recovery Duration by Region")
            st.pyplot(fig)
    else:
        st.warning("No rows have both confirmed_date and released_date — recovery analysis unavailable.")

# ---- TAB 4: REGIONAL IMPACT ----
with tab4:
    if df["region"].notna().sum() > 0 and df["state"].notna().sum() > 0:
        region_summary = df.groupby("region")["state"].value_counts().unstack(fill_value=0)
        region_summary["confirmed_total"] = region_summary.sum(axis=1)
        region_summary = region_summary.sort_values("confirmed_total", ascending=False).head(10)
        cols_to_plot = ["confirmed_total"] + (["released"] if "released" in region_summary.columns else [])
        fig, ax = plt.subplots(figsize=(10, 5))
        region_summary[cols_to_plot].plot(kind="bar", ax=ax)
        ax.set_title("Confirmed vs. Released Cases by Region (Top 10)")
        st.pyplot(fig)
        st.dataframe(region_summary)
    else:
        st.warning("Region or state column not found.")

# ---- TAB 5: REGRESSION MODEL ----
with tab5:
    if len(released) > 0:
        feature_candidates = ["age", "contact_number", "infection_order"]
        usable_features = [c for c in feature_candidates if released[c].notna().sum() >= 20]
        st.write(f"**Usable predictor columns:** {usable_features if usable_features else 'None'}")

        if usable_features:
            model_df = released[usable_features + ["recovery_duration"]].dropna()
            st.write(f"**Rows used for modeling:** {len(model_df)}")

            if len(model_df) >= 20:
                X = model_df[usable_features]
                y = model_df["recovery_duration"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                lr = LinearRegression().fit(X_train, y_train)
                y_pred = lr.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                c1, c2, c3 = st.columns(3)
                c1.metric("R² Score", f"{r2:.3f}")
                c2.metric("MAE (days)", f"{mae:.2f}")
                c3.metric("RMSE (days)", f"{rmse:.2f}")

                st.subheader("Coefficients")
                st.dataframe(pd.Series(lr.coef_, index=X.columns, name="Coefficient"))

                fig, ax = plt.subplots(figsize=(7, 4))
                residuals = y_test - y_pred
                sns.scatterplot(x=y_pred, y=residuals, ax=ax)
                ax.axhline(0, color="red", linestyle="--")
                ax.set_xlabel("Predicted Recovery Duration")
                ax.set_ylabel("Residual")
                ax.set_title("Residual Plot")
                st.pyplot(fig)

                st.subheader("Statistical Significance (OLS Summary)")
                X_sm = sm.add_constant(X)
                ols_model = sm.OLS(y, X_sm).fit()
                st.text(ols_model.summary())
            else:
                st.warning("Not enough complete rows to train a reliable model.")
        else:
            st.warning("No predictor columns have enough data for regression.")
    else:
        st.warning("No recovery duration data available — regression unavailable.")

st.divider()
st.caption("Built for HealthGuard Analytics Pvt. Ltd. — Minor Project Dashboard")
