import torch
import numpy as np

# ============================================================
# MÉTRICAS DE AVALIAÇÃO (Accuracy, Precision, Recall, F1)
# ============================================================

def compute_metrics(y_true, y_pred):
    """
    Calcula as métricas:
    - Acurácia
    - Precisão
    - Recall
    - F1 Score
    """
    # Ajustando y_pred: se for >= 0.5, prediz 1; senão, prediz 0
    y_pred_bin = (y_pred >= 0.5).float()

    # Calculo das métricas
    TP = (y_true * y_pred_bin).sum().item()  # Verdadeiro Positivo
    TN = ((1 - y_true) * (1 - y_pred_bin)).sum().item()  # Verdadeiro Negativo
    FP = ((1 - y_true) * y_pred_bin).sum().item()  # Falso Positivo
    FN = (y_true * (1 - y_pred_bin)).sum().item()  # Falso Negativo

    # Acurácia
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precisão
    precision = TP / (TP + FP + 1e-8)

    # Recall
    recall = TP / (TP + FN + 1e-8)

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, precision, recall, f1
