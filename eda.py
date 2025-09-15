import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
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


def run_full_profile(df: pd.DataFrame):
    """Run ydata-profiling and return HTML as string."""
    profile = ProfileReport(df, explorative=True, minimal=True)
    buffer = io.StringIO()
    profile.to_file(buffer, output_file=None)
    return buffer.getvalue()
