import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from models.neural_network import MLP, DeepMLP
from utils.data_loader import load_wine_dataset, load_breast_cancer_dataset, load_iris_dataset
import argparse
import time
import random
import numpy as np
from models.es import train_es
from models.de import train_de, train_deaw, EDEAdam  # Import wszystkich metod
from models.adam import train_with_adam
# Parser argumentów
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default="experiments/results/history_all_final.json", help="Ścieżka do pliku wynikowego")
parser.add_argument('--repeats', type=int, default=1, help="Ile razy powtórzyć każdy eksperyment z różnymi seedami")
args = parser.parse_args()

output_path = args.output
repeats = args.repeats

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Załaduj dane (wraz ze zbiorami testowymi)
train_loader_wine, valid_loader_wine, test_loader_wine, input_size_wine, output_size_wine = load_wine_dataset(batch_size=32, seed=42)
train_loader_bcw, valid_loader_bcw, test_loader_bcw, input_size_bcw, output_size_bcw = load_breast_cancer_dataset(batch_size=32, seed=42)
train_loader_iris, valid_loader_iris, test_loader_iris, input_size_iris, output_size_iris = load_iris_dataset(batch_size=32, seed=42)

# Definicja modeli dla każdego zbioru danych
models = {
    'wine': {
        'MLP': MLP(input_size=input_size_wine, hidden_size=10, output_size=output_size_wine),
        'DeepMLP': DeepMLP(input_size=input_size_wine, hidden_sizes=[input_size_wine*2, input_size_wine], output_size=output_size_wine)
    },
    'iris': {
        'MLP': MLP(input_size=input_size_iris, hidden_size=input_size_iris*2, output_size=output_size_iris),
        'DeepMLP': DeepMLP(input_size=input_size_iris, hidden_sizes=[input_size_iris*2, input_size_iris], output_size=output_size_iris)
    },
    'bcw': {
        'MLP': MLP(input_size=input_size_bcw, hidden_size=input_size_bcw*2, output_size=output_size_bcw),
        'DeepMLP': DeepMLP(input_size=input_size_bcw, hidden_sizes=[input_size_bcw*2, input_size_bcw], output_size=output_size_bcw)
    }
}

datasets = {
    'wine': (train_loader_wine, valid_loader_wine, test_loader_wine),
    'iris': (train_loader_iris, valid_loader_iris, test_loader_iris),
    'bcw': (train_loader_bcw, valid_loader_bcw, test_loader_bcw)
}

configurations = {
    'ES': [
        {'mu': mu, 'lambda_': lambda_, 'variant_mu_lambda': variant, 'modified_ES': True}
        for mu in [90, 70, 50, 30]
        for lambda_ in [90, 70, 50, 30]
        for variant in [False, True]
    ],
    'DE': [
        {'NP': NP, 'F': F, 'CR': CR, 'max_generations': max_gen, 'initial_lower': -1.0, 'initial_upper': 1.0}
        for NP in [50, 20]
        for F in [0.7, 0.5, 0.3]
        for CR in [0.9, 0.7]
        for max_gen in [150, 100, 50]
    ],
    'DEAW': [
        {'NP': NP, 'F': F, 'CR': CR, 'max_generations': max_gen, 'initial_lower': -1.0, 'initial_upper': 1.0}
        for NP in [70, 50, 30]
        for F in [0.8, 0.5, 0.3]
        for CR in [1.0, 0.9, 0.7]
        for max_gen in [150, 100, 50]
    ],
    'EDEAdam': [
        {'pop_size': pop_size, 'max_evals': max_evals, 'exchange_interval': exchange_interval, 'batch_size': 32}
        for pop_size in [250, 100, 50, 25]
        for max_evals in [1000, 500]
        for exchange_interval in [100, 50, 25, 10, 5]
    ],
    'Adam': [
        {'epochs': epoch, 'learning_rate': 0.001}
        for epoch in [50, 30, 10]
    ]
}


seeds = [random.randint(0, 100000) for _ in range(repeats)]

# Funkcja do trenowania modelu z daną metodą
def train_model(model, train_loader, valid_loader, test_loader, method, config, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'test_loss': [], 'test_accuracy': []}

    if method == 'ES':
        model_trained, history_es = train_es(
            model, train_loader, valid_loader,
            variant_mu_lambda=config['variant_mu_lambda'],
            modified_ES=config['modified_ES'],
            mu=config['mu'],
            lambda_=config['lambda_'],
            max_evals=15000,
            device=device,
            print_metrics=False,
            seed=seed
        )
        # Przekształcenie history_es do zgodnego formatu
        for gen in history_es:
            history['train_loss'].append(gen.get('train_loss', 0))
            history['train_accuracy'].append(gen.get('train_accuracy', 0))
            history['val_loss'].append(gen.get('val_loss', 0))
            history['val_accuracy'].append(gen.get('val_accuracy', 0))
        # Metryki na całym zbiorze testowym
        model_trained.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_trained(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                total_test_correct += (predicted == targets).sum().item()
                total_test_samples += targets.size(0)
        history['test_loss'].append(total_test_loss / total_test_samples)
        history['test_accuracy'].append(100 * total_test_correct / total_test_samples)
        return model_trained, history
    
    elif method == 'DE':
        model_trained, train_history = train_de(
            model, train_loader, valid_loader, device,
            NP=config['NP'],
            F=config['F'],
            CR=config['CR'],
            max_generations=config['max_generations'],
            initial_lower=config['initial_lower'],
            initial_upper=config['initial_upper']
        )
        # Użyj historii z train_de
        for gen in train_history:
            history['train_loss'].append(gen['train_loss'])
            history['train_accuracy'].append(gen['train_accuracy'])
            history['val_loss'].append(gen['val_loss'])
            history['val_accuracy'].append(gen['val_accuracy'])
        # Metryki na całym zbiorze testowym
        model_trained.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_trained(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                total_test_correct += (predicted == targets).sum().item()
                total_test_samples += targets.size(0)
        history['test_loss'].append(total_test_loss / total_test_samples)
        history['test_accuracy'].append(100 * total_test_correct / total_test_samples)
        return model_trained, history
    
    elif method == 'DEAW':
        model_trained, train_history = train_deaw(
            model, train_loader, valid_loader, device,
            NP=config['NP'],
            F=config['F'],
            CR=config['CR'],
            max_generations=config['max_generations'],
            initial_lower=config['initial_lower'],
            initial_upper=config['initial_upper']
        )
        # Użyj historii z train_deaw
        for gen in train_history:
            history['train_loss'].append(gen['train_loss'])
            history['train_accuracy'].append(gen['train_accuracy'])
            history['val_loss'].append(gen['val_loss'])
            history['val_accuracy'].append(gen['val_accuracy'])

        # Metryki na całym zbiorze testowym
        model_trained.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_trained(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                total_test_correct += (predicted == targets).sum().item()
                total_test_samples += targets.size(0)
        history['test_loss'].append(total_test_loss / total_test_samples)
        history['test_accuracy'].append(100 * total_test_correct / total_test_samples)
        return model_trained, history
    
    elif method == 'EDEAdam':
        ede_adam = EDEAdam(
            model,
            pop_size=config['pop_size'],
            max_evals=config['max_evals'],
            exchange_interval=config['exchange_interval'],
            batch_size=config['batch_size']
        )
        model_trained, train_history = ede_adam.run(train_loader, valid_loader, device)  # Dodano device
        # Użyj historii z EDEAdam
        for gen in train_history:
            history['train_loss'].append(gen['train_loss'])
            history['train_accuracy'].append(gen['train_accuracy'])
            history['val_loss'].append(gen['val_loss'])
            history['val_accuracy'].append(gen['val_accuracy'])
        # Metryki na całym zbiorze testowym
        model_trained.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_trained(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                total_test_correct += (predicted == targets).sum().item()
                total_test_samples += targets.size(0)
        history['test_loss'].append(total_test_loss / total_test_samples)
        history['test_accuracy'].append(100 * total_test_correct / total_test_samples)
        return model_trained, history
    
    elif method == 'Adam':
        model_trained, train_history = train_with_adam(
            model, train_loader, valid_loader,
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            device=device,
            print_metrics=False
        )
        # Przepisz historię do formatu jak w innych metodach
        for gen in train_history:
            history['train_loss'].append(gen['train_loss'])
            history['train_accuracy'].append(gen['train_accuracy'])
            history['val_loss'].append(gen['val_loss'])
            history['val_accuracy'].append(gen['val_accuracy'])

        # Metryki na całym zbiorze testowym
        model_trained.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_trained(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_test_loss += loss.item() * inputs.size(0)
                predicted = torch.argmax(outputs, dim=1)
                total_test_correct += (predicted == targets).sum().item()
                total_test_samples += targets.size(0)
        history['test_loss'].append(total_test_loss / total_test_samples)
        history['test_accuracy'].append(100 * total_test_correct / total_test_samples)
        return model_trained, history

    else:
        raise ValueError(f"Nieznana metoda: {method}")


# Funkcje pomocnicze z de.py
def compute_loss(model, inputs, targets):
    outputs = model(inputs)
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(outputs, targets).item()

def compute_accuracy(model, inputs, targets):
    with torch.no_grad():
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).float().sum()
        accuracy = correct / inputs.size(0)
    return accuracy.item()

# Trenowanie i zapisywanie wyników
results = {}
training_idx = 0
start_global_time = time.time()

# Poprawne liczenie liczby treningów
total_trainings = 0
for method in configurations:
    for dataset_name in datasets:
        num_models = len(models[dataset_name])
        num_configs = len(configurations[method])
        total_trainings += num_models * num_configs * len(seeds)

for dataset_name, (train_loader, valid_loader, test_loader) in datasets.items():
    for model_name, model in models[dataset_name].items():
        for method, method_configs in configurations.items():
            for config in method_configs:
                data = {}
                for repeat, seed in enumerate(seeds):
                    trainings_done = training_idx + 1
                    trainings_left = total_trainings - trainings_done
                    print(f"\n=== Powtórka {repeat+1}/{repeats}, seed={seed} ===\n")
                    print(f"Trenowań wykonano: {trainings_done} / {total_trainings} (pozostało: {trainings_left})")
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    print(f'training_idx {training_idx}')
                    print(f'Dataset_name: {dataset_name}')
                    print(f'model_name: {model_name}')
                    print(f'method: {method}')
                    print(f'config: {config}')
                    start_time = time.time()

                    model = model.to(device)
                    model_trained, history = train_model(
                        model, train_loader, valid_loader, test_loader,
                        method=method,
                        config=config,
                        device=device,
                        seed=seed
                    )
                    elapsed_time = time.time() - start_time
                    data[seed] = {'model': model_trained, 'history': history, 'training_time': elapsed_time}
                    print(f"Czas trenowania: {elapsed_time:.2f} sekund")

                key = f"{dataset_name}_{model_name}_{method}_{str(config)}"
                results[key] = data
                
                training_idx += 1

os.makedirs(os.path.dirname(output_path), exist_ok=True)

elapsed_global_time = time.time() - start_global_time
print(f'Eksperymenty trwały łącznie: {elapsed_global_time:.2f} sekund')

# Przygotowanie wyników do serializacji
def tensor_to_float(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    if isinstance(obj, dict):
        return {k: tensor_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_float(x) for x in obj]
    return obj

def model_to_str(model):
    return str(model)

results_serializable = {
    k: {
        seed: {
            'history': tensor_to_float(data['history']),
            'training_time': data['training_time'],
            'model': model_to_str(data['model'])
        }
        for seed, data in v.items()
    }
    for k, v in results.items()
}

with open(output_path, "w") as f:
    json.dump(results_serializable, f, indent=2)

print(f"Historia treningu zapisana do: {output_path}")