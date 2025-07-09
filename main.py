import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset


def extract_bert_embeddings(texts, model, tokenizer, device="cpu", batch_size=32):
    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        embeddings.append(output.cpu())

    return torch.cat(embeddings, dim=0).numpy()


def evaluate_logistic(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    return acc, f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset("ag_news")
    texts = dataset["train"]["text"][:2000]
    labels = dataset["train"]["label"][:2000]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    print("Extracting BERT embeddings...")
    features = extract_bert_embeddings(texts, model, tokenizer, device)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    print("Evaluating baseline logistic regression...")
    acc_orig, f1_orig = evaluate_logistic(X_train, X_test, y_train, y_test)

    print("Applying PCA compression to 100 dimensions...")
    pca = PCA(n_components=100)
    features_pca = pca.fit_transform(features)
    X_train_pca, X_test_pca, _, _ = train_test_split(
        features_pca, labels, test_size=0.2, random_state=42
    )

    print("Evaluating compressed embeddings...")
    acc_pca, f1_pca = evaluate_logistic(X_train_pca, X_test_pca, y_train, y_test)

    results = pd.concat(
        [
            pd.DataFrame(
                [
                    {"Embedding": "Original (768)", "Accuracy": acc_orig, "F1": f1_orig}
                ]
            ),
            pd.DataFrame(
                [
                    {"Embedding": "PCA (100)", "Accuracy": acc_pca, "F1": f1_pca}
                ]
            ),
        ],
        ignore_index=True,
    )

    print("\n=== Evaluation Summary ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
