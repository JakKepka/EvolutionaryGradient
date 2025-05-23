import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()

def train_with_adam(model, train_loader, epochs, learning_rate, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Inicjalizacja paska postępu dla epoki
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Aktualizacja statystyk
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Obliczenie średniej straty i dokładności
            avg_loss = running_loss / total_samples
            accuracy = 100.0 * correct_predictions / total_samples

            # Aktualizacja paska postępu
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Accuracy': f'{accuracy:.2f}%'})

        # Wyświetlenie statystyk po każdej epoce
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
