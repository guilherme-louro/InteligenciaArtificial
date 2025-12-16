import torch
import numpy as np

# ============================================================
# MÉTRICAS DE AVALIAÇÃO (Accuracy, Precision, Recall, F1)
# ============================================================

def compute_metrics(y_true, y_pred):
    """
    Calcula as métricas clássicas para classificação binária:
    - Acurácia
    - Precisão
    - Recall
    - F1 Score

    Parametros:
        y_true  : tensor com os valores verdadeiros (0 ou 1)
        y_pred  : tensor com as previsões do modelo (valores entre 0 e 1)

    Retorna:
        accuracy, precision, recall, f1
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


# ============================================================
# EXEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    # Dados fictícios para testar
    y_true = torch.tensor([1, 0, 1, 1, 0])
    y_pred = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.2])

    accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")