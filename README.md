# Compressed BERT Embeddings for Text Classification

## Description

This project uses a pretrained BERT model to extract sentence embeddings on the AG News dataset and applies **PCA-based dimensionality reduction** to compress them. Classification performance is compared between original and reduced embeddings using Logistic Regression.

## Dataset

- **AG News**: 4-class news topic classification
- Automatically loaded via HuggingFace `datasets`

## Requirements

```bash
pip install transformers datasets scikit-learn pandas
```
## Sample Results
```bash
Extracting BERT embeddings...
Evaluating baseline logistic regression...
Applying PCA compression to 100 dimensions...
Evaluating compressed embeddings...

=== Evaluation Summary ===
      Embedding  Accuracy     F1
 Original (768)    0.9165  0.9160
     PCA (100)    0.8975  0.8971
```
