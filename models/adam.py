import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()

def train_with_adam(model, train_loader, valid_loader, epochs, learning_rate, device='cpu', print_metrics=True):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize training history
    history = []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
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

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        avg_val_loss = val_loss / val_total
        val_accuracy = 100.0 * val_correct / val_total

        # Store statistics in history
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })

        if print_metrics:
            print(f"Epoch {epoch+1}:")
            print(f"  Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"  Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    # Final validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == targets).sum().item()
            val_total += targets.size(0)

    final_val_loss = val_loss / val_total
    final_val_accuracy = 100.0 * val_correct / val_total

    if print_metrics:
        print(f"Final Validation Loss: {final_val_loss:.4f}, Accuracy: {final_val_accuracy:.2f}%")

    return model, history