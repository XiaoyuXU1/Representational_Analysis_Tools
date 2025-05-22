<h1 align="center">Representational Analysis</h1>

[![PyPI version](https://img.shields.io/pypi/v/representational_analysis.svg)](https://pypi.org/project/representational_analysis/)  
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
pip install representational-analysis

````

---
````markdown
## 🛠️ Python API

Import and call the unified function directly:

```python
from representational_toolkit.analysis import run_feature_analysis

query = [
    "The quick brown fox jumps over the lazy dog.",
    "Unlearning LLMs is an active area of research."
]
````

### 1) Fisher Information

```python
run_feature_analysis(
    feature="fim",
    model_reference_path="Qwen/Qwen2.5-7B",
    model_path="./checkpoints/patched",
    query=query,
    output_path="./fim_output",   # directory for per-layer PDFs
    device="cuda",
    batch_size=4,
    num_batches=10,
    max_length=128
)
```

### 2) PCA Shift (Δ PC1 vs. PC2)

```python
run_feature_analysis(
    feature="pca_shift",
    model_reference_path="Qwen/Qwen2.5-7B",
    model_path="./checkpoints/patched",
    query=query,
    output_path="./pca_shift.pdf",  # single PDF
    device="cuda",
    max_length=128
)
```

### 3) PCA Cosine Similarity of PC1

```python
run_feature_analysis(
    feature="pca_sim",
    model_reference_path="Qwen/Qwen2.5-7B",
    model_path="./checkpoints/patched",
    query=query,
    output_path="./pca_sim.pdf",    # single PDF
    device="cuda",
    max_length=128
)
```

### 4) Layer-wise CKA

```python
run_feature_analysis(
    feature="cka",
    model_reference_path="Qwen/Qwen2.5-7B",
    model_path="./checkpoints/patched",
    query=query,
    output_path="./cka.pdf",        # single PDF
    device="cuda",
    batch_size=4,
    num_batches=10,
    max_length=128
)
```


---

## 📁 Project Structure

```
representational_analysis/
├── pyproject.toml
└── src/
    └── representational_toolkit/
        ├── __init__.py
        ├── analysis.py         # `run_analysis()` entry point
        ├── fim_analysis.py     # Fisher Information
        ├── pca_shift_analysis.py
        ├── pca_sim_analysis.py
        └── cka_analysis.py     # Layer-wise CKA
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -m "Add foo"`)
4. Push to your branch (`git push origin feature/foo`)
5. Open a Pull Request

Please follow PEP8 style and add tests for new functionality.

---

## 📜 License

This project is licensed under MIT License. See [LICENSE](LICENSE) for details.

```
```
