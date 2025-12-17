# RAG+BERT Code Classification System

A Retrieval-Augmented Generation (RAG) system using BERT for multi-label classification of coding problems. This system combines contrastive learning, FAISS-based retrieval, and cross-encoding to accurately classify programming exercises into multiple categories.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a sophisticated multi-label classification system for coding problems using:

- **Contrastive Learning**: Learn meaningful embeddings for code exercises
- **FAISS Retrieval**: Fast similarity search for relevant examples
- **Cross-Encoding**: Fine-grained relevance scoring
- **Multi-Label Classification**: Classify exercises into 8 categories

### Target Categories

- `math` - Mathematical reasoning and formulas
- `graphs` - Graph theory and network structures
- `strings` - String manipulation and text processing
- `number theory` - Prime numbers, divisibility, modular arithmetic
- `trees` - Tree data structures and traversal
- `geometry` - Geometric reasoning and spatial relationships
- `games` - Game theory and optimal strategies
- `probabilities` - Probability theory and statistics

## ✨ Features

- **Two-Stage Architecture**: Combines retrieval and re-ranking for optimal performance
- **Contrastive Learning**: Learns discriminative embeddings using triplet loss
- **FAISS Integration**: Efficient similarity search over large datasets
- **Cross-Encoder Re-ranking**: Fine-grained relevance scoring
- **Configurable Pipeline**: Easy configuration via YAML files
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Interactive Prediction**: Command-line interface for testing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Exercise                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Contrastive Encoder   │
         │    (BERT-based)        │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   FAISS Retrieval      │
         │   (Top-K Similar)      │
         └────────────┬───────────┘
                      │
         ┌────────────┴───────────┐
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│ Tag Aggregation │    │  Cross-Encoder   │
│  from Neighbors │    │   Re-ranking     │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌────────────────────────┐
         │   Final Predictions    │
         │  (Weighted Ensemble)   │
         └────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd BERT_RAG
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare your data**
- Place your JSON dataset files in `data/code_classification_dataset_cleaned/`
- Update `config.yaml` with the correct data path

## 📁 Project Structure

```
BERT_RAG/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── dataset.py                # PyTorch Dataset classes
│   ├── models.py                 # Neural network models
│   ├── trainer.py                # Training utilities
│   ├── predictor.py              # Prediction and retrieval
│   └── utils.py                  # Evaluation and visualization
│
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── predict.py                # Prediction script
│
├── notebook/                     # Jupyter notebooks
│   └── RAG+BERT.ipynb           # Original notebook
│
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize the system:

```yaml
# Model settings
model:
  base_model_name: "bert-base-uncased"
  max_length: 256
  projection_dim: 128

# Training settings
training:
  batch_size: 4
  learning_rate: 0.00002
  num_epochs_contrastive: 3
  num_epochs_cross: 3

# Retrieval settings
retrieval:
  top_k: 10
  alpha: 0.6  # Weight for retrieval
  beta: 0.4   # Weight for cross-encoder
```

## 🚀 Usage

### 1. Training

Train both the contrastive encoder and cross-encoder:

```bash
python scripts/train.py
```

This will:
- Load and preprocess the data
- Train the contrastive encoder
- Create FAISS index
- Train the cross-encoder
- Save all models and artifacts

**Output files:**
- `outputs/retrieval_model/best_contrastive_encoder.pt`
- `outputs/retrieval_model/best_cross_encoder.pt`
- `outputs/retrieval_model/faiss_index.bin`
- `outputs/retrieval_model/data_splits.pkl`

### 2. Evaluation

Evaluate the trained models:

```bash
python scripts/evaluate.py
```

This will:
- Load trained models
- Predict on validation and test sets
- Find optimal thresholds
- Generate evaluation metrics
- Create visualization plots

**Output files:**
- `outputs/retrieval_model/final_results.json`
- `outputs/retrieval_model/final_results_comparison.png`
- `outputs/retrieval_model/optimal_thresholds.json`

### 3. Prediction

Make predictions on new exercises:

```bash
python scripts/predict.py
```

This provides:
- Example prediction demonstration
- Interactive mode for custom inputs
- Detailed prediction scores and similar exercises

### Example Prediction

```python
from src.config import Config
from src.predictor import RetrievalAugmentedPredictor
# ... load models ...

exercise = """
Given an array of integers, find the maximum sum of a contiguous subarray.
[SEP] Input: An array of integers nums
[SEP] Output: Return the maximum sum
"""

result = predictor.predict_single(exercise)
print(result['final_scores'])
```

## 🧠 Model Details

### Contrastive Encoder

- **Base Model**: BERT-base-uncased (110M parameters)
- **Architecture**: BERT + Projection Head
- **Loss**: Contrastive Loss + Triplet Loss
- **Output**: 128-dimensional normalized embeddings

### Cross-Encoder

- **Base Model**: BERT-base-uncased
- **Architecture**: BERT + Binary Classifier
- **Loss**: Binary Cross-Entropy
- **Output**: Relevance score [0, 1]

### Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Scheduler**: Cosine Annealing
- **Gradient Clipping**: Max norm 1.0
- **Batch Size**: 4 (adjustable)

## 📊 Results

### Performance Metrics

Example results on test set:

| Metric | Score |
|--------|-------|
| F1 Macro | 0.XXX |
| F1 Micro | 0.XXX |
| Precision Macro | 0.XXX |
| Recall Macro | 0.XXX |
| Exact Match | 0.XXX |

### Per-Tag Performance

| Tag | Precision | Recall | F1-Score |
|-----|-----------|--------|----------|
| math | 0.XXX | 0.XXX | 0.XXX |
| graphs | 0.XXX | 0.XXX | 0.XXX |
| strings | 0.XXX | 0.XXX | 0.XXX |
| ... | ... | ... | ... |

*Note: Run evaluation to get actual metrics*

## 🔧 Advanced Usage

### Custom Configuration

Create a custom config file:

```python
from src.config import Config

config = Config("my_config.yaml")
```

### Batch Prediction

```python
import pandas as pd
from src.predictor import RetrievalAugmentedPredictor

# Load your data
df = pd.read_csv("exercises.csv")

# Predict
scores_df = predictor.predict_batch(df)
```

### Fine-tuning

Adjust hyperparameters in `config.yaml`:

```yaml
training:
  batch_size: 8  # Increase if you have more GPU memory
  learning_rate: 3e-5  # Adjust learning rate
  num_epochs_contrastive: 5  # More epochs
```

## 📈 Monitoring

Training progress is automatically saved:

- Loss curves: `outputs/retrieval_model/*_training_history.png`
- Checkpoints: `outputs/retrieval_model/*.pt`
- Logs: Console output

## 🐛 Troubleshooting

### Out of Memory

Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 2
```

### Slow Training

- Use GPU if available
- Reduce `max_length` in config
- Use smaller model (e.g., `distilbert-base-uncased`)

### Poor Performance

- Increase training epochs
- Adjust `alpha` and `beta` weights
- Increase `top_k` for retrieval
- Check data quality and distribution




---

**Note**: This is a research project. Performance may vary depending on your dataset and configuration.
