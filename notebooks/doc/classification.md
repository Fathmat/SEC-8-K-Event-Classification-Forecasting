# Multi-Label Classification of SEC 8-K Filings using Fine-Tuned BERT

This module classifies SEC 8-K filings into **multiple relevant event categories** using a fine-tuned BERT model. Each filing can belong to **1, 2, or 3 categories** from a predefined taxonomy of 14 (one version of this mutliclassification has 33) event types.



## Problem Formulation

We treat this as a **multi-label classification** task.

![BERT multi-label classification formula](notebooks/doc/problem formula.png)
Let:
- \( x_i \) be the input filing (text)
- \( \mathbf{y}_i = [y_{i1}, y_{i2}, ..., y_{iC}] \) be the target binary label vector for \( C \) categories, where:
  - \( y_{ij} = 1 \) if the \( j^{th} \) category applies to the \( i^{th} \) document, else 0



## Model Architecture

We use a pretrained BERT model (`bert-base-uncased`) and add a **sigmoid-activated linear output layer** for multi-label prediction:

\[
\hat{\mathbf{y}} = \sigma(W \cdot h_{[CLS]} + b)
\]

Where:
- \( h_{[CLS]} \) is the final hidden state of the [CLS] token
- \( W \in \mathbb{R}^{C \times H} \), \( b \in \mathbb{R}^{C} \)
- \( \sigma \) is the sigmoid function applied element-wise



## Training Details

- **Model**: `bert-base-uncased`
- **Loss**: Binary Cross-Entropy Loss (BCE) for each label:
  \[
  \mathcal{L} = -\sum_{j=1}^{C} \left[ y_j \log(\hat{y}_j) + (1 - y_j) \log(1 - \hat{y}_j) \right]
  \]
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup and decay
- **Thresholding**: Labels are assigned if \( \hat{y}_j > \tau \), where \( \tau \) is a tunable threshold (e.g. 0.5)



## Evaluation Metrics

Standard accuracy isn't enough in multi-label settings. We use:
- **Micro / Macro F1-score**
- **Precision / Recall**
- **Hamming Loss**
- **Label ranking loss** (optional for interpretability)



## Input / Output Format

Input: Full 8-K filing text or extracted summary  
Output: A set of relevant event categories, such as:
- `Senior Personnel Change`
- `Litigation and Lawsuits`
- `Financial Activities`

Example output for one filing:
```json
["Senior Personnel Change", "Information Disclosure", "Document Updates"]
