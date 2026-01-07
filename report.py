from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
from datetime import datetime

def create_docx_report(title: str, executive_summary: str, eda_summary: dict, model_summary: str, charts: list):
    """
    Create a professional Word report.
    
    Args:
        title: Report title
        executive_summary: AI-generated executive summary
        eda_summary: Dictionary with EDA statistics
        model_summary: String representation of model results
        charts: List of chart file paths (optional)
    
    Returns:
        Path to the generated .docx file
    """
    doc = Document()
    
    # Add title
    title_para = doc.add_heading(title, level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add date
    date_para = doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_page_break()
    
    # Executive Summary Section
    doc.add_heading('Executive Summary', level=1)
    doc.add_paragraph(executive_summary)
    
    doc.add_page_break()
    
    # EDA Section
    doc.add_heading('Exploratory Data Analysis', level=1)
    doc.add_heading('Dataset Overview', level=2)
    
    table = doc.add_table(rows=len(eda_summary) + 1, cols=2)
    table.style = 'Light Grid Accent 1'
    
    # Header row
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Metric'
    header_cells[1].text = 'Value'
    
    # Data rows
    for i, (key, value) in enumerate(eda_summary.items(), start=1):
        row_cells = table.rows[i].cells
        row_cells[0].text = str(key)
        row_cells[1].text = str(value)
    
    doc.add_page_break()
    
    # Model Results Section
    doc.add_heading('Model Performance', level=1)
    doc.add_paragraph(model_summary)
    
    # Add charts if provided
    if charts:
        doc.add_page_break()
        doc.add_heading('Visualizations', level=1)
        for chart_path in charts:
            if os.path.exists(chart_path):
                doc.add_picture(chart_path, width=Inches(5))
    
    # Save document
    filename = f"report_{title.replace(' ', '_')}.docx"
    doc.save(filename)
    
    return filename