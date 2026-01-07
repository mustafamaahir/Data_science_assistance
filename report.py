from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def add_page_break(doc):
    """Add page break to document."""
    doc.add_page_break()


def add_hyperlink(paragraph, text, url):
    """Add a hyperlink to a paragraph."""
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    
    new_run = OxmlElement('w:r')
    r_pr = OxmlElement('w:rPr')
    
    new_run.append(r_pr)
    new_run.text = text
    hyperlink.append(new_run)
    
    paragraph._p.append(hyperlink)
    
    return hyperlink


def create_table_from_dict(doc, data_dict, title=None):
    """Create a formatted table from dictionary."""
    if title:
        doc.add_heading(title, level=3)
    
    table = doc.add_table(rows=len(data_dict) + 1, cols=2)
    table.style = 'Light Grid Accent 1'
    
    # Header
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Metric'
    header_cells[1].text = 'Value'
    
    # Data rows
    for i, (key, value) in enumerate(data_dict.items(), start=1):
        row_cells = table.rows[i].cells
        row_cells[0].text = str(key)
        row_cells[1].text = str(value)
    
    doc.add_paragraph()  # Spacing
    return table


def create_metrics_chart(metrics: dict, problem_type: str, output_path: str):
    """Create a visual chart of model metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter metrics for visualization
    viz_metrics = {k: v for k, v in metrics.items() 
                   if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']
                   and isinstance(v, (int, float))}
    
    if viz_metrics:
        metric_names = list(viz_metrics.keys())
        metric_values = list(viz_metrics.values())
        
        colors = sns.color_palette('viridis', len(metric_names))
        bars = ax.barh(metric_names, metric_values, color=colors)
        
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Metrics ({problem_type.capitalize()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(metric_values) * 1.1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {width:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    return None


def create_feature_importance_chart(feature_importance_df, output_path: str):
    """Create feature importance visualization."""
    if feature_importance_df is None or feature_importance_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = feature_importance_df.head(20)
    
    colors = sns.color_palette('rocket', len(top_features))
    bars = ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {width:.4f}',
               ha='left', va='center', fontsize=9)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_confusion_matrix_chart(confusion_matrix, class_labels, output_path: str):
    """Create confusion matrix visualization."""
    cm = np.array(confusion_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_comprehensive_report(
    df,
    executive_summary: str,
    preprocessing_log: list,
    imputation_decisions: dict,
    model_result: dict,
    eda_charts: list,
    generated_code: str
):
    """
    Create a comprehensive Word report with all analyses.
    
    Args:
        df: Original DataFrame
        executive_summary: AI-generated summary
        preprocessing_log: List of preprocessing steps
        imputation_decisions: Dict of imputation methods
        model_result: Model training results
        eda_charts: List of EDA chart paths
        generated_code: Generated Python code
    
    Returns:
        Path to generated report
    """
    doc = Document()
    
    # ========== TITLE PAGE ==========
    title = doc.add_heading('Machine Learning Project Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph('Comprehensive Data Science Analysis')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_format = subtitle.runs[0]
    subtitle_format.font.size = Pt(16)
    subtitle_format.font.color.rgb = RGBColor(70, 70, 150)
    
    doc.add_paragraph()
    
    date_para = doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    
    # Dataset info on title page
    info_para = doc.add_paragraph()
    info_para.add_run(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n").bold = True
    info_para.add_run(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    add_page_break(doc)
    
    # ========== TABLE OF CONTENTS ==========
    doc.add_heading('Table of Contents', level=1)
    
    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Exploratory Data Analysis",
        "4. Data Preprocessing & Cleaning",
        "5. Model Development",
        "6. Model Performance & Evaluation",
        "7. Feature Importance Analysis",
        "8. Reproducible Code",
        "9. Conclusions & Recommendations"
    ]
    
    for item in toc_items:
        doc.add_paragraph(item, style='List Number')
    
    add_page_break(doc)
    
    # ========== 1. EXECUTIVE SUMMARY ==========
    doc.add_heading('1. Executive Summary', level=1)
    doc.add_paragraph(executive_summary)
    
    add_page_break(doc)
    
    # ========== 2. DATASET OVERVIEW ==========
    doc.add_heading('2. Dataset Overview', level=1)
    
    doc.add_heading('2.1 Dataset Characteristics', level=2)
    
    dataset_info = {
        'Total Records': f"{df.shape[0]:,}",
        'Total Features': df.shape[1],
        'Numeric Features': len(df.select_dtypes(include='number').columns),
        'Categorical Features': len(df.select_dtypes(exclude='number').columns),
        'Total Missing Values': f"{df.isna().sum().sum():,}",
        'Missing Percentage': f"{(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%",
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'Duplicate Rows': f"{df.duplicated().sum():,}"
    }
    
    create_table_from_dict(doc, dataset_info)
    
    doc.add_heading('2.2 Feature Types Distribution', level=2)
    doc.add_paragraph(
        f"The dataset contains {len(df.select_dtypes(include='number').columns)} numeric features "
        f"and {len(df.select_dtypes(exclude='number').columns)} categorical features. "
        f"This distribution provides insights into the nature of the data and informs "
        f"our preprocessing strategy."
    )
    
    add_page_break(doc)
    
    # ========== 3. EXPLORATORY DATA ANALYSIS ==========
    doc.add_heading('3. Exploratory Data Analysis', level=1)
    
    doc.add_paragraph(
        "Comprehensive exploratory data analysis was conducted to understand data distributions, "
        "relationships between features, and identify potential data quality issues."
    )
    
    # Add EDA charts
    if eda_charts:
        for i, chart_path in enumerate(eda_charts, 1):
            if os.path.exists(chart_path):
                doc.add_heading(f'3.{i} Visualization', level=2)
                doc.add_picture(chart_path, width=Inches(6))
                doc.add_paragraph()  # Spacing
    else:
        doc.add_paragraph("Note: EDA visualizations were not generated. Please run EDA step in the application.")
    
    add_page_break(doc)
    
    # ========== 4. DATA PREPROCESSING ==========
    doc.add_heading('4. Data Preprocessing & Cleaning', level=1)
    
    doc.add_heading('4.1 Preprocessing Strategy', level=2)
    doc.add_paragraph(
        "A comprehensive preprocessing pipeline was implemented including MCAR (Missing Completely At Random) "
        "testing, intelligent imputation method selection, and feature engineering."
    )
    
    doc.add_heading('4.2 Imputation Methods Applied', level=2)
    
    if imputation_decisions:
        imputation_summary = {}
        method_counts = {}
        
        for col, method in imputation_decisions.items():
            method_counts[method] = method_counts.get(method, 0) + 1
        
        imputation_summary = {
            'Total Columns Processed': len(imputation_decisions),
            'KNN Imputation': method_counts.get('knn', 0),
            'Regression Imputation': method_counts.get('regression', 0),
            'Simple Imputation': method_counts.get('simple', 0) + method_counts.get('mode', 0),
            'Rows Dropped': method_counts.get('drop', 0),
            'No Imputation Needed': method_counts.get('none', 0)
        }
        
        create_table_from_dict(doc, imputation_summary)
    
    doc.add_heading('4.3 Detailed Preprocessing Log', level=2)
    
    if preprocessing_log:
        for log_entry in preprocessing_log[:30]:  # Limit to first 30 entries
            p = doc.add_paragraph(log_entry, style='List Bullet')
            p.paragraph_format.left_indent = Inches(0.25)
    else:
        doc.add_paragraph("Preprocessing log not available.")
    
    add_page_break(doc)
    
    # ========== 5. MODEL DEVELOPMENT ==========
    doc.add_heading('5. Model Development', level=1)
    
    if model_result:
        model_name = model_result.get('model_name', 'Unknown')
        
        doc.add_heading('5.1 Model Selection', level=2)
        doc.add_paragraph(
            f"The selected model for this analysis is **{model_name}**. "
            f"This choice was based on the problem type, dataset characteristics, "
            f"and performance requirements."
        )
        
        doc.add_heading('5.2 Hyperparameters', level=2)
        
        if 'best_params' in model_result:
            create_table_from_dict(doc, model_result['best_params'], "Optimized Hyperparameters")
        else:
            doc.add_paragraph("Default hyperparameters were used.")
        
        doc.add_heading('5.3 Training Details', level=2)
        
        training_info = {
            'Training Set Size': f"{model_result.get('X_train_shape', ('N/A', 'N/A'))[0]:,} samples",
            'Test Set Size': f"{model_result.get('X_test_shape', ('N/A', 'N/A'))[0]:,} samples",
            'Number of Features': model_result.get('X_train_shape', ('N/A', 'N/A'))[1],
            'Model Type': model_name
        }
        
        create_table_from_dict(doc, training_info)
    
    add_page_break(doc)
    
    # ========== 6. MODEL PERFORMANCE ==========
    doc.add_heading('6. Model Performance & Evaluation', level=1)
    
    if model_result and 'metrics' in model_result:
        metrics = model_result['metrics']
        problem_type = 'classification' if 'accuracy' in metrics else 'regression'
        
        doc.add_heading('6.1 Performance Metrics', level=2)
        
        # Create metrics visualization
        temp_dir = 'temp_report_charts'
        os.makedirs(temp_dir, exist_ok=True)
        
        metrics_chart_path = os.path.join(temp_dir, 'metrics_chart.png')
        chart_path = create_metrics_chart(metrics, problem_type, metrics_chart_path)
        
        if chart_path and os.path.exists(chart_path):
            doc.add_picture(chart_path, width=Inches(6))
            doc.add_paragraph()
        
        # Metrics table
        clean_metrics = {k: f"{v:.4f}" if isinstance(v, float) else v 
                        for k, v in metrics.items() 
                        if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'residuals']}
        
        create_table_from_dict(doc, clean_metrics)
        
        # Classification-specific sections
        if problem_type == 'classification':
            doc.add_heading('6.2 Confusion Matrix', level=2)
            
            if 'confusion_matrix' in metrics and 'classification_report' in metrics:
                cm = metrics['confusion_matrix']
                report = metrics['classification_report']
                
                class_labels = [k for k in report.keys() 
                              if k not in ['accuracy', 'macro avg', 'weighted avg']]
                
                cm_chart_path = os.path.join(temp_dir, 'confusion_matrix.png')
                cm_path = create_confusion_matrix_chart(cm, class_labels, cm_chart_path)
                
                if cm_path and os.path.exists(cm_path):
                    doc.add_picture(cm_path, width=Inches(5.5))
                    doc.add_paragraph()
            
            doc.add_heading('6.3 Classification Report', level=2)
            
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                
                # Create detailed report table
                doc.add_paragraph("Per-class performance metrics:")
                
                report_data = []
                for class_name, class_metrics in report.items():
                    if isinstance(class_metrics, dict):
                        report_data.append({
                            'Class': class_name,
                            'Precision': f"{class_metrics.get('precision', 0):.3f}",
                            'Recall': f"{class_metrics.get('recall', 0):.3f}",
                            'F1-Score': f"{class_metrics.get('f1-score', 0):.3f}",
                            'Support': class_metrics.get('support', 0)
                        })
                
                if report_data:
                    table = doc.add_table(rows=len(report_data) + 1, cols=5)
                    table.style = 'Light Grid Accent 1'
                    
                    # Headers
                    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
                    for i, header in enumerate(headers):
                        table.rows[0].cells[i].text = header
                    
                    # Data
                    for i, row_data in enumerate(report_data, 1):
                        for j, (key, value) in enumerate(row_data.items()):
                            table.rows[i].cells[j].text = str(value)
        
        # Regression-specific sections
        else:
            doc.add_heading('6.2 Prediction Analysis', level=2)
            doc.add_paragraph(
                f"The model achieved an R² score of {metrics.get('r2_score', 0):.4f}, "
                f"indicating {'excellent' if metrics.get('r2_score', 0) > 0.9 else 'good' if metrics.get('r2_score', 0) > 0.7 else 'moderate'} "
                f"predictive performance."
            )
    
    else:
        doc.add_paragraph("Model performance metrics not available. Please train a model first.")
    
    add_page_break(doc)
    
    # ========== 7. FEATURE IMPORTANCE ==========
    doc.add_heading('7. Feature Importance Analysis', level=1)
    
    if model_result and model_result.get('feature_importance') is not None:
        doc.add_paragraph(
            "Feature importance analysis reveals which variables have the strongest "
            "influence on the model's predictions. Understanding feature importance "
            "helps in model interpretation and feature selection for future iterations."
        )
        
        feature_importance_df = model_result['feature_importance']
        
        # Create visualization
        fi_chart_path = os.path.join(temp_dir, 'feature_importance.png')
        fi_path = create_feature_importance_chart(feature_importance_df, fi_chart_path)
        
        if fi_path and os.path.exists(fi_path):
            doc.add_picture(fi_path, width=Inches(6))
            doc.add_paragraph()
        
        # Top features table
        doc.add_heading('7.1 Top 15 Most Important Features', level=2)
        
        top_15 = feature_importance_df.head(15)
        
        table = doc.add_table(rows=len(top_15) + 1, cols=3)
        table.style = 'Light Grid Accent 1'
        
        # Headers
        table.rows[0].cells[0].text = 'Rank'
        table.rows[0].cells[1].text = 'Feature'
        table.rows[0].cells[2].text = 'Importance Score'
        
        # Data
        for i, (idx, row) in enumerate(top_15.iterrows(), 1):
            table.rows[i].cells[0].text = str(i)
            table.rows[i].cells[1].text = row['Feature']
            table.rows[i].cells[2].text = f"{row['Importance']:.6f}"
    
    else:
        doc.add_paragraph(
            "Feature importance analysis is not available for this model type or "
            "the model has not been trained yet."
        )
    
    add_page_break(doc)
    
    # ========== 8. REPRODUCIBLE CODE ==========
    doc.add_heading('8. Reproducible Code', level=1)
    
    doc.add_paragraph(
        "The following Python code can be used to reproduce this analysis. "
        "This code includes all preprocessing steps, model training, and evaluation."
    )
    
    if generated_code:
        code_para = doc.add_paragraph(generated_code)
        code_para.style = 'Normal'
        
        # Format as code (monospace font)
        for run in code_para.runs:
            run.font.name = 'Courier New'
            run.font.size = Pt(9)
    else:
        doc.add_paragraph(
            "Reproducible code was not generated. Please generate code in the Report section "
            "of the application."
        )
    
    add_page_break(doc)
    
    # ========== 9. CONCLUSIONS ==========
    doc.add_heading('9. Conclusions & Recommendations', level=1)
    
    doc.add_heading('9.1 Key Findings', level=2)
    doc.add_paragraph(
        "• Comprehensive data preprocessing including MCAR testing and intelligent imputation\n"
        "• Model training and evaluation completed with detailed performance metrics\n"
        "• Feature importance analysis provides insights into key predictive variables\n"
        "• Complete reproducible code provided for implementation"
    )
    
    doc.add_heading('9.2 Recommendations', level=2)
    doc.add_paragraph(
        "Based on this analysis, the following recommendations are suggested:\n\n"
        "1. Model Performance: Review the performance metrics and determine if they meet "
        "business requirements. Consider ensemble methods or model stacking if performance "
        "needs improvement.\n\n"
        "2. Feature Engineering: Explore additional feature engineering opportunities based "
        "on domain knowledge and the feature importance analysis.\n\n"
        "3. Data Collection: Address missing data issues at the source if possible to improve "
        "data quality for future modeling efforts.\n\n"
        "4. Monitoring: Implement model monitoring in production to track performance degradation "
        "and trigger retraining when necessary.\n\n"
        "5. Explainability: Consider implementing SHAP or LIME for individual prediction "
        "explanations if model interpretability is critical."
    )
    
    doc.add_heading('9.3 Next Steps', level=2)
    doc.add_paragraph(
        "• Deploy the model to production environment\n"
        "• Set up automated retraining pipeline\n"
        "• Implement monitoring and alerting\n"
        "• Document model limitations and assumptions\n"
        "• Plan for model updates and improvements"
    )
    
    # ========== FOOTER ==========
    doc.add_paragraph()
    footer_para = doc.add_paragraph()
    footer_para.add_run("─" * 80)
    footer_para.add_run(
        f"\n\nReport generated by Data Science Assistant\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Save document
    filename = f"comprehensive_ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(filename)
    
    return filename