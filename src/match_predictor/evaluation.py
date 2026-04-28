import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate(model_name, y_true, y_pred, le):
    class_names = le.classes_
    acc = accuracy_score(y_true, y_pred)

    print(f"\n--- {model_name} | Accuracy: {acc:.4f} ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"Confusion matrix:\n{cm_df}\n")

    return acc

