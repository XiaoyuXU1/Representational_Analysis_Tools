# Representational Analysis

[![PyPI version](https://img.shields.io/pypi/v/representational-analysis.svg)](https://pypi.org/project/representational-analysis/)  
[![Python Version](https://img.shields.io/pypi/pyversions/representational-analysis)]  
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
pip install representational_analysis

# Install editable from GitHub
git clone https://github.com/yourusername/representational_analysis.git
cd representational_analysis
pip install -e .
