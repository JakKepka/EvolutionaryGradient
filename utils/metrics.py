from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch 

def evaluate_model(model, data_loader, device='cpu'):
    """
    Ewaluacja modelu na zbiorze danych.

    Args:
        model: Wytrenowany model PyTorch.
        data_loader: DataLoader zawierający dane testowe.
        device: Urządzenie ('cpu' lub 'cuda').

    Returns:
        Dictionary z metrykami: accuracy, precision, recall, f1.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, dtype=torch.float64)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(np.array(all_labels), np.array(all_preds))

def compute_metrics(y_true, y_pred, average='macro'):
    """
    Oblicza metryki klasyfikacji.

    Args:
        y_true: Prawdziwe etykiety.
        y_pred: Przewidywane etykiety.
        average: Typ uśredniania ('macro', 'micro', 'weighted').

    Returns:
        Dictionary z metrykami: accuracy, precision, recall, f1.
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    return metrics
