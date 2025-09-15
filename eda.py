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
    if not df.select_dtypes(include='number').empty:
        fig, ax = plt.subplots()
        df.hist(ax=ax)
        figs.append(fig)

    # Correlation heatmap
    if df.select_dtypes(include='number').shape[1] > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        figs.append(fig)

    return summary_cards, figs



def run_full_profile(df):
    report = sv.analyze(df)
    report_path = "sweetviz_report.html"
    report.show_html(report_path, open_browser=False)
    with open(report_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=800, scrolling=True)
