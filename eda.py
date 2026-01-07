import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from ydata_profiling import ProfileReport

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def quick_eda(df: pd.DataFrame):
    """Generate quick EDA stats and plots."""
    if df is None or df.empty:
        return {}, []
    
    summary_cards = {
        "Shape": f"{df.shape[0]:,} rows × {df.shape[1]} columns",
        "Missing": f"{df.isna().sum().sum():,}",
        "Numeric": len(df.select_dtypes(include='number').columns),
        "Categorical": len(df.select_dtypes(exclude='number').columns),
        "Memory": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }

    figs = []

    # Distribution of numeric features
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        n_cols = min(len(numeric_df.columns), 12)  # Limit to 12 for readability
        n_rows = (n_cols + 2) // 3
        
        fig1, axes1 = plt.subplots(
            nrows=n_rows,
            ncols=3,
            figsize=(15, 4 * n_rows)
        )
        axes1 = axes1.flatten() if n_rows > 1 else [axes1] if n_cols == 1 else axes1
        
        for i, col in enumerate(numeric_df.columns[:n_cols]):
            numeric_df[col].hist(ax=axes1[i], bins=30, edgecolor='black', alpha=0.7)
            axes1[i].set_title(f'{col}\n(mean={numeric_df[col].mean():.2f})', fontsize=10)
            axes1[i].set_xlabel('')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes1)):
            axes1[j].axis('off')
        
        plt.tight_layout()
        figs.append(fig1)

    # Correlation heatmap
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, 
            mask=mask,
            annot=True if corr.shape[0] <= 10 else False,
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax2
        )
        ax2.set_title('Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        figs.append(fig2)

    # Missing data visualization
    missing_series = df.isna().sum().sort_values(ascending=False).head(20)
    missing_series = missing_series[missing_series > 0]
    
    if not missing_series.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        colors = sns.color_palette('viridis', len(missing_series))
        bars = ax3.barh(missing_series.index, missing_series.values, color=colors)
        ax3.set_xlabel('Missing Values Count', fontsize=12)
        ax3.set_title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            percentage = (width / len(df)) * 100
            ax3.text(width, bar.get_y() + bar.get_height()/2, 
                    f' {int(width)} ({percentage:.1f}%)',
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        figs.append(fig3)

    return summary_cards, figs


def create_eda_charts(df: pd.DataFrame, output_dir='eda_charts'):
    """
    Create comprehensive EDA charts for report inclusion.
    
    Returns:
        List of file paths to generated charts
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    chart_paths = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Dataset Overview
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Data types distribution
    type_counts = df.dtypes.value_counts()
    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Data Types Distribution', fontsize=12, fontweight='bold')
    
    # Missing values percentage
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(15)
    if not missing_pct.empty and missing_pct.max() > 0:
        missing_pct.plot(kind='barh', ax=axes[0, 1], color='coral')
        axes[0, 1].set_xlabel('Missing %')
        axes[0, 1].set_title('Top 15 Columns by Missing %', fontsize=12, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Missing Values', 
                       ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')
    
    # Numeric vs Categorical
    num_count = len(df.select_dtypes(include='number').columns)
    cat_count = len(df.select_dtypes(exclude='number').columns)
    axes[1, 0].bar(['Numeric', 'Categorical'], [num_count, cat_count], 
                   color=['steelblue', 'lightcoral'])
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Feature Type Distribution', fontsize=12, fontweight='bold')
    for i, v in enumerate([num_count, cat_count]):
        axes[1, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Dataset size info
    info_text = f"""
    Rows: {df.shape[0]:,}
    Columns: {df.shape[1]}
    Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    Duplicates: {df.duplicated().sum():,}
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', 
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Dataset Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path1 = os.path.join(output_dir, f'overview_{timestamp}.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    chart_paths.append(path1)
    
    # 2. Numeric distributions (box plots)
    numeric_cols = df.select_dtypes(include='number').columns[:12]
    if len(numeric_cols) > 0:
        n_rows = (len(numeric_cols) + 2) // 3
        fig2, axes2 = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes2 = axes2.flatten() if n_rows > 1 else [axes2]
        
        for i, col in enumerate(numeric_cols):
            sns.boxplot(y=df[col], ax=axes2[i], color='skyblue')
            axes2[i].set_title(f'{col}', fontsize=10)
            axes2[i].set_ylabel('')
        
        for j in range(i + 1, len(axes2)):
            axes2[j].axis('off')
        
        plt.tight_layout()
        path2 = os.path.join(output_dir, f'numeric_distributions_{timestamp}.png')
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(path2)
    
    # 3. Categorical distributions (top categories)
    cat_cols = df.select_dtypes(exclude='number').columns[:8]
    if len(cat_cols) > 0:
        n_rows = (len(cat_cols) + 1) // 2
        fig3, axes3 = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        axes3 = axes3.flatten() if n_rows > 1 else [axes3]
        
        for i, col in enumerate(cat_cols):
            top_values = df[col].value_counts().head(10)
            top_values.plot(kind='barh', ax=axes3[i], color='lightgreen')
            axes3[i].set_title(f'{col} (Top 10)', fontsize=10)
            axes3[i].set_xlabel('Count')
            axes3[i].invert_yaxis()
        
        for j in range(i + 1, len(axes3)):
            axes3[j].axis('off')
        
        plt.tight_layout()
        path3 = os.path.join(output_dir, f'categorical_distributions_{timestamp}.png')
        plt.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(path3)
    
    # 4. Correlation heatmap (high-res)
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] > 1:
        fig4, ax4 = plt.subplots(figsize=(14, 12))
        corr = numeric_df.corr()
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True if corr.shape[0] <= 15 else False,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax4
        )
        ax4.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        path4 = os.path.join(output_dir, f'correlation_matrix_{timestamp}.png')
        plt.savefig(path4, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(path4)
    
    # 5. Statistical summary table (as image)
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig5, ax5 = plt.subplots(figsize=(14, max(6, len(numeric_df.columns) * 0.4)))
        ax5.axis('tight')
        ax5.axis('off')
        
        stats_df = numeric_df.describe().T
        stats_df = stats_df.round(2)
        
        table = ax5.table(
            cellText=stats_df.values,
            rowLabels=stats_df.index,
            colLabels=stats_df.columns,
            cellLoc='right',
            rowLoc='right',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style row labels
        for i in range(1, len(stats_df) + 1):
            table[(i, -1)].set_facecolor('#f0f0f0')
            table[(i, -1)].set_text_props(weight='bold')
        
        ax5.set_title('Statistical Summary of Numeric Features', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        path5 = os.path.join(output_dir, f'statistical_summary_{timestamp}.png')
        plt.savefig(path5, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(path5)
    
    return chart_paths


def run_full_profile(df: pd.DataFrame):
    """Generate full profiling report using ydata-profiling."""
    try:
        
        profile = ProfileReport(
            df, 
            title="Comprehensive Data Profiling Report",
            explorative=True,
            minimal=False
        )
        report_html = profile.to_html()
        
        return report_html
    
    except ImportError:
        return "<h1>Error: ydata-profiling not installed</h1><p>Install with: pip install ydata-profiling</p>"
    except Exception as e:
        return f"<h1>Error generating profile</h1><p>{str(e)}</p>"