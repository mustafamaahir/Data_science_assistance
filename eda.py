import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quick_eda(df: pd.DataFrame):
    """Generate quick EDA stats and plots."""
    if df is None or df.empty:
        return {}, []
    
    summary_cards = {
        "Shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "Missing values": int(df.isna().sum().sum()),
        "Numeric cols": len(df.select_dtypes(include='number').columns),
        "Categorical cols": len(df.select_dtypes(exclude='number').columns)
    }

    figs = []

    # Distribution of numeric features
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig1, axes1 = plt.subplots(
            nrows=(len(numeric_df.columns) + 2) // 3,
            ncols=3,
            figsize=(12, 4 * ((len(numeric_df.columns) + 2) // 3))
        )
        if len(numeric_df.columns) == 1:
            axes1 = [axes1]
        else:
            axes1 = axes1.flatten() if hasattr(axes1, 'flatten') else axes1
        
        for i, col in enumerate(numeric_df.columns):
            numeric_df[col].hist(ax=axes1[i], bins=30)
            axes1[i].set_title(col)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes1)):
            axes1[j].axis('off')
        
        plt.tight_layout()
        figs.append(fig1)

    # Correlation heatmap (only numeric columns)
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap="coolwarm", ax=ax2, square=True)
        ax2.set_title('Correlation Matrix')
        plt.tight_layout()
        figs.append(fig2)

    # Missingness bar for top 20 missing columns
    missing_series = df.isna().sum().sort_values(ascending=False).head(20)
    missing_series = missing_series[missing_series > 0]
    
    if not missing_series.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=missing_series.values, y=missing_series.index, ax=ax3, palette='viridis')
        ax3.set_xlabel('Missing values count')
        ax3.set_title('Top Missing Columns')
        plt.tight_layout()
        figs.append(fig3)

    return summary_cards, figs


def run_full_profile(df: pd.DataFrame):
    """Generate full profiling report using ydata-profiling."""
    try:
        from ydata_profiling import ProfileReport
        
        profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
        report_html = profile.to_html()
        
        return report_html
    
    except ImportError:
        return "<h1>Error: ydata-profiling not installed</h1><p>Install with: pip install ydata-profiling</p>"
    except Exception as e:
        return f"<h1>Error generating profile</h1><p>{str(e)}</p>"