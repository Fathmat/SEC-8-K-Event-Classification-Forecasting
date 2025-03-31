# 📊 Multi-Label Classification of SEC 8-K Filings using Fine-Tuned BERT

This module classifies SEC 8-K filings into **multiple relevant event categories** using a fine-tuned BERT model. Each filing can belong to **1, 2, or 3 categories** from a predefined taxonomy of 33 event types.

---

## 🔍 Problem Formulation

We treat this as a **multi-label classification** task.

## Model Formula

![BERT multi-label classification formula](notebooks/doc/problem formula.png)


---

## 🧠 Model Architecture

We use a pretrained BERT model (`bert-base-uncased`) and add a **sigmoid-activated linear output layer** for multi-label prediction:

![BERT multi-label classification formula2](notebooks/doc/model architecture.png)
---

## ⚙️ Training Details

- **Model**: `bert-base-uncased`
- **Loss**: Binary Cross-Entropy Loss (BCE) for each label:
![BERT multi-label classification formula2](notebooks/doc/Training details.png)
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup and decay
- **Thresholding**: Labels are assigned if \( \hat{y}_j > \tau \), where \( \tau \) is a tunable threshold (e.g. 0.5)

---

## 🧪 Evaluation Metrics

Standard accuracy isn't enough in multi-label settings. We use:
- **Micro / Macro F1-score**
- **Precision / Recall**
- **Hamming Loss**
- **Label ranking loss** (optional for interpretability)

---

## 📁 Input / Output Format

Input: Full 8-K filing text or extracted summary  
Output: A set of relevant event categories, such as:
- `Senior Personnel Change`
- `Litigation and Lawsuits`
- `Financial Activities`

Example output for one filing:
```json
["Senior Personnel Change", "Information Disclosure", "Document Updates"]
