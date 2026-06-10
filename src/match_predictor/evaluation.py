import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss


def evaluate(model_name, y_true, y_pred, le, y_proba=None):
    class_names = le.classes_
    acc = accuracy_score(y_true, y_pred)

    header = f"\n--- {model_name} | Accuracy: {acc:.4f}"
    if y_proba is not None:
        loss = log_loss(y_true, y_proba, labels=range(len(class_names)))
        header += f" | Log loss: {loss:.4f}"
    print(header + " ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"Confusion matrix:\n{cm_df}\n")

    return acc

