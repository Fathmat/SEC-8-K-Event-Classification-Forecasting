# 📄 SEC 8-K Event Classification & Forecasting

This project uses NLP models to classify and forecast events in SEC 8-K filings — turning unstructured disclosure data into structured signals that can support financial decision-making.



## 🚀 Project Overview

The goal is to extract insights from 8-K reports by:
- Classifying filings into event types (e.g., Senior Personnel Change, M&A, Financial Activities).
- Forecasting what event is likely to occur next based on historical sequences.



## 🧠 Methodology

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


