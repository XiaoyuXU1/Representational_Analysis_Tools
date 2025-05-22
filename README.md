# Feature_analysis_to# Feature Analyzer

[![PyPI version](https://img.shields.io/pypi/v/feature_analyzer.svg)](https://pypi.org/project/feature_analyzer/)  
[![Python Version](https://img.shields.io/pypi/pyversions/feature_analyzer)]  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A unified toolkit for analyzing representational changes in large language models (LLMs).  
Supports:

- **Fisher Information Matrix** histograms  
- **PCA shift** (Δ PC1 vs. PC2)  
- **PCA cosine similarity** of principal components  
- **Layer-wise CKA** (Centered Kernel Alignment)  

---

## 🔧 Installation

```bash
# From PyPI
pip install feature_analyzer

# Or install editable from source
git clone https://github.com/yourusername/feature_analyzer.git
cd feature_analyzer
pip install -e .
ols
