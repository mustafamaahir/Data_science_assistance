import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import streamlit as st
import io

def quick_eda(df: pd.DataFrame):
    """Generate quick EDA stats and plots."""
    summary_cards = {
        "Shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "Missing values": df.isna().sum().sum(),
        "Numeric cols": len(df.select_dtypes(include='number').columns),
        "Categorical cols": len(df.select_dtypes(exclude='number').columns)
    }

    figs = []

    # Distribution of numeric features
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        numeric_df.hist(ax=ax1)
        figs.append(fig1)

    # Correlation heatmap (only numeric columns)
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax2)
        figs.append(fig2)

    # Missingness bar for top 20 missing columns
    missing_series = df.isna().sum().sort_values(ascending=False).head(20)
    if not missing_series.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        sns.barplot(x=missing_series.values, y=missing_series.index, ax=ax3)
        ax3.set_xlabel('Missing values')
        figs.append(fig3)

    return summary_cards, figs

def run_full_profile(df):
    report = sv.analyze(df)
    report_path = "sweetviz_report.html"
    report.show_html(report_path, open_browser=False)
    with open(report_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=800, scrolling=True)
