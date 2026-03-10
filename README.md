# AI Data Quality Agent

## 🚀 Overview

An intelligent data quality assessment tool that leverages Large Language Models (LLMs) to automatically analyze datasets and provide actionable insights. This project demonstrates the practical application of AI in data engineering workflows, combining traditional data quality checks with advanced LLM-powered analysis.

## 🎯 Project Goals

This project was built as a portfolio piece to showcase:
- **AI Integration**: Practical implementation of LLM APIs for data analysis
- **Big Data Skills**: Understanding of data quality principles and profiling
- **Cloud Computing**: Free-tier Google Colab for accessible ML workflows
- **Career Transition**: Bridging big data engineering with AI/ML capabilities

## ✨ Features

### 1. Comprehensive Data Profiling
- **Basic Statistics**: Mean, median, mode, standard deviation
- **Data Type Analysis**: Automatic detection and classification
- **Missing Value Detection**: Identifies and quantifies null values
- **Uniqueness Analysis**: Detects duplicate records and unique constraints

### 2. AI-Powered Insights
- **Anomaly Detection**: LLM identifies unusual patterns and outliers
- **Quality Scoring**: Professional-grade assessment (0-100 scale)
- **Root Cause Analysis**: Explains why quality issues occur
- **Actionable Recommendations**: Specific steps to improve data quality

### 3. Multi-Model Support
- **OpenRouter API**: Primary LLM integration (free tier)
- **Hugging Face Inference**: Fallback option for API limitations
- **Flexible Architecture**: Easy to extend with additional models

## 🏗️ Architecture

```
Data Input → Profiling Engine → Quality Checks → LLM Analysis → Comprehensive Report
     ↓              ↓                  ↓              ↓                ↓
   CSV/DF      Statistics        Validation      AI Insights      JSON/Text
```

## 🛠️ Technology Stack

- **Language**: Python 3.x
- **Platform**: Google Colab (free tier)
- **Libraries**: 
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
  - `requests`: API interactions
- **AI/ML**: 
  - OpenRouter API (nVidia Nemotron model)
  - Hugging Face Inference API
- **Cost**: 100% Free (no paid services)

## 📋 Prerequisites

- Google account (for Colab)
- OpenRouter API key (free tier available at https://openrouter.ai/)
- Basic understanding of Python and data analysis

## 🚀 Getting Started

### 1. Setup

1. Open the notebook in Google Colab
2. Get your free OpenRouter API key:
   ```
   Visit: https://openrouter.ai/
   Sign up → API Keys → Create new key
   ```
3. Set your API key in the notebook:
   ```python
   OPENROUTER_API_KEY = "YOUR_API_KEY_HERE"
   ```

### 2. Running the Analysis

```python
# Step 1: Create sample data
data = create_sample_data()

# Step 2: Profile the data
quality_findings = analyze_data_quality(data)

# Step 3: Get AI insights
diagnosis = get_ai_diagnosis(quality_findings)

# Step 4: View results
print(diagnosis)
```

### 3. Using Your Own Data

```python
import pandas as pd

# Load your CSV
df = pd.read_csv('your_data.csv')

# Run analysis
quality_findings = analyze_data_quality(df)
diagnosis = get_ai_diagnosis(quality_findings)
```

## 📊 Sample Output

The tool generates a comprehensive report including:

```json
{
  "quality_score": 72,
  "severity": "MEDIUM",
  "key_issues": [
    "Missing values in critical columns (8%)",
    "Potential duplicates detected (12 records)"
  ],
  "recommendations": [
    "Implement data validation at source",
    "Add deduplication logic in ETL pipeline"
  ]
}
```

## 🎓 Use Cases

### 1. Data Engineering
- **ETL Validation**: Check data quality before/after transformations
- **Pipeline Monitoring**: Automated quality alerts
- **Schema Evolution**: Detect unexpected changes

### 2. Business Intelligence
- **Report Accuracy**: Ensure dashboard data is reliable
- **Trend Analysis**: Identify data anomalies affecting metrics

### 3. Machine Learning
- **Training Data Quality**: Validate datasets before model training
- **Feature Engineering**: Discover data issues early

## 🔍 Why LLMs for Data Quality?

Traditional data quality tools excel at statistical checks but lack context. LLMs add:

1. **Contextual Understanding**: Interprets what numbers mean for business
2. **Pattern Recognition**: Identifies subtle anomalies humans might miss
3. **Natural Language Output**: Technical insights in plain English
4. **Adaptive Analysis**: No need to pre-define all quality rules

## 🆓 Cost Optimization

This project uses **100% free services**:

- ✅ Google Colab: Free GPU/TPU access
- ✅ OpenRouter: Free tier (nvidia/nemotron-3-nano model)
- ✅ Hugging Face: Free inference API
- ✅ No database costs: In-memory processing
- ✅ No storage costs: Works with temporary data



## 📚 Learning Outcomes

This project demonstrates proficiency in:

- ✅ **Python Programming**: Clean, modular code
- ✅ **Data Analysis**: Statistical profiling with pandas/numpy
- ✅ **API Integration**: RESTful API consumption
- ✅ **AI/ML Concepts**: Prompt engineering, LLM integration
- ✅ **Big Data Principles**: Data quality dimensions (completeness, accuracy, etc.)
- ✅ **Cloud Computing**: Google Colab workflow

## 🛣️ Future Enhancements

- [ ] Add visualization dashboard (Plotly/Streamlit)
- [ ] Support for multiple file formats (Excel, JSON, Parquet)
- [ ] Integration with data catalogs (Apache Atlas)
- [ ] Real-time streaming data quality
- [ ] Custom rule engine for domain-specific checks
- [ ] Export to data quality frameworks (Great Expectations)

## 📖 Related Technologies

**Similar to**:
- Amazon Deequ (Spark-based data quality)
- Great Expectations (Python data testing)
- Apache Griffin (big data quality)

**But different because**:
- Uses AI for contextual analysis
- Lightweight (no Spark required)
- Natural language output

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use this for learning and portfolio purposes.




---

**⭐ If this project helped you, please star the repository!**
