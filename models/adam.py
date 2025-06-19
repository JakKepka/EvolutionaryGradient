import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()

def train_with_adam(model, train_loader, valid_loader, epochs, learning_rate, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Trening
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device, dtype=torch.float64), targets.to(device)

            # Zerowanie gradientów
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            avg_loss = running_loss / total_samples
            accuracy = 100.0 * correct_predictions / total_samples

            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Accuracy': f'{accuracy:.2f}%'})

        print(f"Epoch {epoch+1} completed. Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")

        # Walidacja
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device, dtype=torch.float64), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        avg_val_loss = val_loss / val_total
        val_accuracy = 100.0 * val_correct / val_total

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")
