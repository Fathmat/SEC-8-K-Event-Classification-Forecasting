# 📄 SEC 8-K Event Classification & Forecasting

This project uses NLP models to classify and forecast events in SEC 8-K filings — turning unstructured disclosure data into structured signals that can support financial decision-making.



## 🚀 Project Overview

The goal is to extract insights from 8-K reports by:
- Classifying filings into event types (e.g., Senior Personnel Change, Share Repurchase, Financial Activities).
- Forecasting what event is likely to occur next based on historical sequences.



## Methodology

### 1. Classification
- Compared **fuzzy matching** vs. **modern NLP models** (e.g., BERT, gpt-4o-mini).
- Achieved **+7% accuracy improvement** with BERT over the baseline.

### 2. Forecasting
- Trained sequence models like **GRU with Attention** and **T5** to predict next event types.
- Framed forecasting as a sequence-to-sequence task using labeled 8-K data from 2020–2024.



## 📊 Results

- Improved classification accuracy to **79%** using BERT.
- Constructed event-based portfolios (e.g., based on Senior Personnel Change) to test financial impact.
- Positioned as a proof-of-concept for event-driven trading strategies or risk monitoring.



## 🛠 Tech Stack

`Python` | `Transformers` | `BERT` | `T5` | `Pandas` | `Scikit-learn` | `Plotly` | `Bloomberg Terminal` | `LLMs` | `GRU` | `Attention Mechanisms`

---

## 📂 Repo Structure
```bash
.
├── Data/               # Preprocessed 8-K filings
├── notebooks/          # Exploratory and modeling notebooks
├── models/                # Classification and forecasting modules
├── requirements.txt    # Environment setup
└── README.md
```
## 📈 Why This Matters

SEC 8-K filings contain rich signals that are often underused.  
This project builds tools to extract those insights in a scalable and structured way — empowering analysts, investors, and automated systems.

## About Me

My name is **Fathmat** — aspiring quant & NLP explorer.  
📫 [Connect on LinkedIn](www.linkedin.com/in/fathmat-bakayoko-30715024a)

Note: Due to file size and storage limitations, full datasets and some models are hosted on Lehigh’s high-performance computing cluster and are not included here. The repo contains a high-level overview, key scripts, and some lightweight results.
