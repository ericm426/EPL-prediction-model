import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss


def brier_score(y_true, y_proba):
    """Multiclass Brier score — lower is better (perfect = 0, random ≈ 0.67 for 3 classes)."""
    n_classes = y_proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def rps(y_onehot, probs):
    """Mean ranked probability score (multiclass), lower = better. Classes must be in outcome order."""
    cum_prob = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum((cum_prob - cum_actual) ** 2, axis=1) / (probs.shape[1] - 1)))


RESULT_ORDER = ["HOME_TEAM", "DRAW", "AWAY_TEAM"]


def score_probs(results, probs):
    """
    Scores H/D/A probabilities against actual results.
    results: array-like of 'HOME_TEAM'/'DRAW'/'AWAY_TEAM'
    probs:   (n, 3) array with columns in RESULT_ORDER
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_true = pd.Series(results).map({r: i for i, r in enumerate(RESULT_ORDER)}).values
    y_onehot = np.eye(3)[y_true]
    return {
        "accuracy": float((probs.argmax(axis=1) == y_true).mean()),
        "log_loss": float(log_loss(y_true, probs, labels=range(3))),
        "brier":    brier_score(y_true, probs),
        "rps":      rps(y_onehot, probs),
    }


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
