# Advanced Data Science Assistant

A comprehensive, AI-powered machine learning pipeline builder with intelligent preprocessing, automated model selection, and professional reporting.

## ✨ Features

### 1. 📤 Smart Data Upload
- Support for CSV and Excel files
- Automatic encoding detection
- Instant data preview and statistics

### 2. 📊 Comprehensive EDA
- Quick statistical analysis
- Automated visualizations (distributions, correlations, missing data)
- Full profiling reports with ydata-profiling
- High-resolution charts for reporting

### 3. 🔧 Intelligent Preprocessing
- **MCAR Testing**: Statistical tests to determine if data is missing completely at random
- **Smart Imputation**: 
  - Compares KNN vs Regression imputation
  - Uses RMSE/accuracy to select best method
  - Configurable k-neighbors slider
- **Automated Decision Making**:
  - < 5% missing + MCAR → Drop rows
  - \> 5% missing → Compare imputation methods
- **Detailed Logging**: Complete audit trail of all preprocessing decisions

### 4. 🎯 Advanced Feature Selection
- Manual multi-select with checkboxes
- **AI-Suggested Features**: Groq API analyzes features and recommends important ones
- Feature importance visualization
- Domain knowledge integration

### 5. 🤖 Model Training & Evaluation
- Support for multiple algorithms:
  - Classification: RandomForest, LogisticRegression, XGBoost, SVM
  - Regression: RandomForest, LinearRegression, XGBoost, SVR
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive metrics:
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
  - Regression: RMSE, MAE, R², Residual plots
- Feature importance analysis

### 6. 📄 Professional Reporting
- **AI-Generated Executive Summary** (via Groq)
- **Table of Contents** with page breaks
- **Embedded Visualizations**:
  - EDA charts
  - Model performance metrics
  - Feature importance
  - Confusion matrices
  - ROC curves
  - Residual plots
- **Reproducible Python Code**: AI generates complete notebook-ready code
- Professional Word document (.docx) export

### 7. 💻 Code Generation
- **Groq-powered code generation**
- Complete data loading, preprocessing, and modeling pipeline
- All visualizations included
- Ready to run in Jupyter notebooks

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/data-science-assistant.git
cd data-science-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API key**

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Or set environment variable:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

4. **Run the application**
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your repository
- Add secrets in Settings → Secrets:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## 📖 Usage Guide

### Workflow

1. **Upload** → Load your CSV/Excel dataset
2. **EDA & Profiling** → Explore data, generate visualizations
3. **Preprocess** → 
   - Select target column
   - Configure MCAR testing
   - Set imputation parameters (k-neighbors)
   - Run comprehensive preprocessing
   - Review detailed log
4. **Modeling** →
   - Select features (manual or AI-suggested)
   - Choose model and hyperparameters
   - Enable tuning if desired
   - Train and evaluate
5. **Report** →
   - Generate AI executive summary
   - Generate reproducible Python code
   - Create comprehensive Word report

### Configuration Options

#### Preprocessing
- **Drop threshold**: % of missing data in target to trigger row deletion (default: 5%)
- **KNN neighbors**: Number of neighbors for KNN imputation (slider: 3-15)
- **Scale numeric**: Apply StandardScaler to numeric features
- **Skip MCAR**: Skip statistical testing for faster processing

#### Modeling
- **Hyperparameter tuning**: Enable RandomizedSearchCV
- **Tuning iterations**: Number of random parameter combinations to try
- **CV folds**: Number of cross-validation folds

## 🏗️ Project Structure

```
data-science-assistant/
├── app.py                    # Main Streamlit application
├── eda.py                    # EDA functions and chart generation
├── preprocessing.py          # MCAR testing, smart imputation
├── models.py                 # Model training and evaluation
├── report.py                 # Comprehensive report generation
├── llm.py                    # Groq API integration
├── code_generator.py         # AI code generation functions
├── requirements.txt          # Python dependencies
├── .env                      # API keys (local only, gitignored)
└── README.md                 # This file
```

## 🔑 Getting Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and add to `.env` or Streamlit secrets

## 📊 Supported Datasets

- **Format**: CSV, XLSX, XLS
- **Size**: Up to 200MB (Streamlit Cloud default)
- **Features**: Any mix of numeric and categorical columns
- **Missing data**: Any percentage (intelligently handled)

## 🎯 Key Advantages

1. **Intelligent Imputation**: Automatically tests and selects best imputation method
2. **MCAR Testing**: Statistical rigor in handling missing data
3. **AI Integration**: Groq API for insights, summaries, and code generation
4. **Professional Output**: Publication-ready Word reports with TOC
5. **Reproducibility**: Generate complete Python code for any analysis
6. **User-Friendly**: No coding required, intuitive interface

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'docx'`
```bash
pip install python-docx
```

**Issue**: `ydata-profiling` conflicts
```bash
pip install seaborn<0.13
```

**Issue**: Groq API timeout
- Reduce `max_tokens` parameter
- Check internet connection
- Verify API key is valid

**Issue**: Out of memory
- Process smaller datasets
- Reduce number of features
- Skip full profiling report

## 📝 Example Workflow

```python
# 1. Upload your dataset
# 2. Run EDA
# 3. Preprocess with these settings:
#    - Drop threshold: 5%
#    - KNN neighbors: 5
#    - Scale numeric: Yes
#    - Skip MCAR: No (for rigorous analysis)
# 4. Select features (use AI suggestions)
# 5. Train RandomForest with tuning
# 6. Generate report with code
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use for commercial or personal projects

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **ydata-profiling** for comprehensive EDA
- **scikit-learn** for ML algorithms
- **Streamlit** for the amazing framework

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Email: mustafamaahir@gmail.com

## 🎯 Future Enhancements

- [ ] Deep learning model support
- [ ] Time series analysis
- [ ] AutoML integration
- [ ] PDF report generation
- [ ] Real-time model monitoring
- [ ] API endpoint for predictions
- [ ] Multi-language support

---

**Built using Streamlit and Groq AI**