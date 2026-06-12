import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss


def brier_score(y_true, y_proba):
    """Multiclass Brier score — lower is better (perfect = 0, random ≈ 0.67 for 3 classes)."""
    n_classes = y_proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def evaluate(model_name, y_true, y_pred, le, y_proba=None):
    class_names = le.classes_
    acc = accuracy_score(y_true, y_pred)

    header = f"\n--- {model_name} | Accuracy: {acc:.4f}"
    if y_proba is not None:
        loss = log_loss(y_true, y_proba, labels=range(len(class_names)))
        bs = brier_score(y_true, y_proba)
        header += f" | Log loss: {loss:.4f} | Brier: {bs:.4f}"
    print(header + " ---")

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"Confusion matrix:\n{cm_df}\n")

    return acc
