import matplotlib.pyplot as plt

# Funkcja do wizualizacji historii trenowania
def plot_training_history(history, title):
    generations = [entry['generation'] for entry in history]
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_loss'] for entry in history]
    train_accs = [entry['train_accuracy'] for entry in history]
    val_accs = [entry['val_accuracy'] for entry in history]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(generations, train_losses, label='Train Loss')
    plt.plot(generations, val_losses, label='Validation Loss')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(generations, train_accs, label='Train Accuracy')
    plt.plot(generations, val_accs, label='Validation Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.show()


# # Wizualizacja wyników
# for key, result in results.items():
#     plot_training_history(result['history'], title=key)
