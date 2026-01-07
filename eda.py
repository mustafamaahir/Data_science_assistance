import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

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
        n_cols = min(len(numeric_df.columns), 12)
        n_rows = (n_cols + 2) // 3
        
        fig1 = plt.figure(figsize=(15, 4 * n_rows))
        
        for i, col in enumerate(numeric_df.columns[:n_cols], 1):
            ax = fig1.add_subplot(n_rows, 3, i)
            numeric_df[col].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'{col}\n(mean={numeric_df[col].mean():.2f})', fontsize=10)
            ax.set_xlabel('')
        
        plt.tight_layout()
        figs.append(fig1)

    # Correlation heatmap
    if numeric_df.shape[1] > 1:
        fig2 = plt.figure(figsize=(12, 10))
        ax2 = fig2.add_subplot(111)
        
        corr = numeric_df.corr()
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
        fig3 = plt.figure(figsize=(12, 6))
        ax3 = fig3.add_subplot(111)
        
        colors = sns.color_palette('viridis', len(missing_series))
        bars = ax3.barh(missing_series.index, missing_series.values, color=colors)
        ax3.set_xlabel('Missing Values Count', fontsize=12)
        ax3.set_title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        
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
    fig1 = plt.figure(figsize=(14, 10))
    
    # Data types distribution
    ax1 = fig1.add_subplot(2, 2, 1)
    type_counts = df.dtypes.value_counts()
    ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Data Types Distribution', fontsize=12, fontweight='bold')
    
    # Missing values percentage
    ax2 = fig1.add_subplot(2, 2, 2)
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(15)
    if not missing_pct.empty and missing_pct.max() > 0:
        missing_pct.plot(kind='barh', ax=ax2, color='coral')
        ax2.set_xlabel('Missing %')
        ax2.set_title('Top 15 Columns by Missing %', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Missing Values', 
                ha='center', va='center', fontsize=14)
        ax2.axis('off')
    
    # Numeric vs Categorical
    ax3 = fig1.add_subplot(2, 2, 3)
    num_count = len(df.select_dtypes(include='number').columns)
    cat_count = len(df.select_dtypes(exclude='number').columns)
    ax3.bar(['Numeric', 'Categorical'], [num_count, cat_count], 
            color=['steelblue', 'lightcoral'])
    ax3.set_ylabel('Count')
    ax3.set_title('Feature Type Distribution', fontsize=12, fontweight='bold')
    for i, v in enumerate([num_count, cat_count]):
        ax3.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Dataset size info
    ax4 = fig1.add_subplot(2, 2, 4)
    info_text = f"""
    Rows: {df.shape[0]:,}
    Columns: {df.shape[1]}
    Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    Duplicates: {df.duplicated().sum():,}
    """
    ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Dataset Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path1 = os.path.join(output_dir, f'overview_{timestamp}.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    chart_paths.append(path1)
    
    # 2. Numeric distributions (box plots)
    numeric_cols = df.select_dtypes(include='number').columns[:12]
    if len(numeric_cols) > 0:
        n_rows = (len(numeric_cols) + 2) // 3
        fig2 = plt.figure(figsize=(15, 4 * n_rows))
        
        for idx, col in enumerate(numeric_cols, 1):
            ax = fig2.add_subplot(n_rows, 3, idx)
            sns.boxplot(y=df[col], ax=ax, color='skyblue')
            ax.set_title(f'{col}', fontsize=10)
            ax.set_ylabel('')
        
        plt.tight_layout()
        path2 = os.path.join(output_dir, f'numeric_distributions_{timestamp}.png')
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        chart_paths.append(path2)
    
    # 3. Categorical distributions
    cat_cols = df.select_dtypes(exclude='number').columns[:8]
    if len(cat_cols) > 0:
        n_rows = (len(cat_cols) + 1) // 2
        fig3 = plt.figure(figsize=(14, 4 * n_rows))
        
        for idx, col in enumerate(cat_cols, 1):
            ax = fig3.add_subplot(n_rows, 2, idx)
            top_values = df[col].value_counts().head(10)
            
            # Use matplotlib directly - no pandas plotting
            y_pos = np.arange(len(top_values))
            ax.barh(y_pos, top_values.values, color='lightgreen')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_values.index, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f'{col} (Top 10)', fontsize=10)
            ax.set_xlabel('Count')
        
        plt.tight_layout()
        path3 = os.path.join(output_dir, f'categorical_distributions_{timestamp}.png')
        plt.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        chart_paths.append(path3)
    
    # 4. Correlation heatmap
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] > 1:
        fig4 = plt.figure(figsize=(14, 12))
        ax4 = fig4.add_subplot(111)
        
        corr = numeric_df.corr()
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
        plt.close(fig4)
        chart_paths.append(path4)
    
    # 5. Statistical summary table
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        fig5 = plt.figure(figsize=(14, max(6, len(numeric_df.columns) * 0.4)))
        ax5 = fig5.add_subplot(111)
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
        
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(stats_df) + 1):
            table[(i, -1)].set_facecolor('#f0f0f0')
            table[(i, -1)].set_text_props(weight='bold')
        
        ax5.set_title('Statistical Summary of Numeric Features', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        path5 = os.path.join(output_dir, f'statistical_summary_{timestamp}.png')
        plt.savefig(path5, dpi=300, bbox_inches='tight')
        plt.close(fig5)
        chart_paths.append(path5)
    
    return chart_paths


def run_full_profile(df: pd.DataFrame):
    """
    Generate a custom HTML profiling report (ydata-profiling replacement).
    """
    try:
        # Create comprehensive HTML report manually
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profiling Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #7f8c8d;
                    text-transform: uppercase;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Data Profiling Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Dataset Overview</h2>
                <div class="metric">
                    <div class="metric-value">{df.shape[0]:,}</div>
                    <div class="metric-label">Rows</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">Columns</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df.duplicated().sum():,}</div>
                    <div class="metric-label">Duplicate Rows</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df.isna().sum().sum():,}</div>
                    <div class="metric-label">Missing Values</div>
                </div>
                
                <h2>Column Statistics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Type</th>
                            <th>Non-Null Count</th>
                            <th>Null Count</th>
                            <th>Null %</th>
                            <th>Unique Values</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df) * 100)
            html_content += f"""
                        <tr>
                            <td><strong>{col}</strong></td>
                            <td>{df[col].dtype}</td>
                            <td>{df[col].notna().sum():,}</td>
                            <td>{null_count:,}</td>
                            <td>{null_pct:.2f}%</td>
                            <td>{df[col].nunique():,}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
                
                <h2>Numeric Column Statistics</h2>
        """
        
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            stats = numeric_df.describe().T
            html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Mean</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>50%</th>
                            <th>75%</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for col in stats.index:
                html_content += f"""
                        <tr>
                            <td><strong>{col}</strong></td>
                            <td>{stats.loc[col, 'mean']:.2f}</td>
                            <td>{stats.loc[col, 'std']:.2f}</td>
                            <td>{stats.loc[col, 'min']:.2f}</td>
                            <td>{stats.loc[col, '25%']:.2f}</td>
                            <td>{stats.loc[col, '50%']:.2f}</td>
                            <td>{stats.loc[col, '75%']:.2f}</td>
                            <td>{stats.loc[col, 'max']:.2f}</td>
                        </tr>
                """
            html_content += """
                    </tbody>
                </table>
            """
        else:
            html_content += "<p>No numeric columns found.</p>"
        
        html_content += """
                <h2>Categorical Column Top Values</h2>
        """
        
        cat_df = df.select_dtypes(exclude='number')
        if not cat_df.empty:
            for col in cat_df.columns[:10]:  # Show first 10 categorical columns
                top_values = df[col].value_counts().head(10)
                html_content += f"""
                <h3>{col}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Value</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for value, count in top_values.items():
                    pct = (count / len(df) * 100)
                    html_content += f"""
                        <tr>
                            <td>{value}</td>
                            <td>{count:,}</td>
                            <td>{pct:.2f}%</td>
                        </tr>
                    """
                html_content += """
                    </tbody>
                </table>
                """
        else:
            html_content += "<p>No categorical columns found.</p>"
        
        html_content += """
                <h2>💡 Recommendations</h2>
                <ul>
                    <li>Use the "Generate Comprehensive EDA Charts" button for visual analysis</li>
                    <li>Check columns with high missing percentages for data quality issues</li>
                    <li>Review unique value counts to identify potential categorical features</li>
                    <li>Proceed to preprocessing to handle missing values and prepare for modeling</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    except Exception as e:
        return f"""
        <div style='padding: 20px; background-color: #fff3cd; border-radius: 10px;'>
            <h1 style='color: #856404;'>⚠️ Error generating profile</h1>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please use the "Generate Comprehensive EDA Charts" button instead.</p>
        </div>
        """