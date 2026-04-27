import torch

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def attack_success_rate(outputs, labels, trigger_label):
    # Placeholder for attack success calculation
    return 0.0