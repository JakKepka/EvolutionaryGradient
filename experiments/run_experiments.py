import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import json
from models.neural_network import MLP, DeepMLP
from utils.data_loader import load_wine_dataset, load_breast_cancer_dataset, load_iris_dataset
import argparse
import time
import random 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default="experiments/results/history_all.json", help="Ścieżka do pliku wynikowego")
parser.add_argument('--repeats', type=int, default=2, help="Ile razy powtórzyć każdy eksperyment z różnymi seedami")
args = parser.parse_args()

output_path = args.output
repeats = args.repeats


# Ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Zakładam, że funkcja train_es jest zdefiniowana gdzie indziej
# Funkcja ta zwraca model_trained, best_fitness, history
from models.es import train_es

seed=42

# Załaduj dane wine
train_loader_wine, valid_loader_wine, test_loader_wine, input_size_wine, output_size_wine = load_wine_dataset(batch_size=32, seed=seed)

# Załaduj dane bcw
train_loader_bcw, valid_loader_bcw, test_loader_bcw, input_size_bcw, output_size_bcw = load_breast_cancer_dataset(batch_size=32, seed=seed)

# Załaduj dane iris
train_loader_iris, valid_loader_iris, test_loader_iris, input_size_iris, output_size_iris = load_iris_dataset(batch_size=32, seed=seed)

# Konfiguracje ES do przetestowania
configurations = [
    {'mu': 30, 'lambda_': 30, 'variant': 'modified-ES'},
    {'mu': 50, 'lambda_': 50, 'variant': 'modified-ES'},
    {'mu': 30, 'lambda_': 30, 'variant': 'mu-lambda'},
    {'mu': 50, 'lambda_': 50, 'variant': 'mu-lambda'},
]

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
    'wine': (train_loader_wine, valid_loader_wine),
    'iris': (train_loader_iris, valid_loader_iris),
    'bcw': (train_loader_bcw, valid_loader_bcw)
}

seeds = [random.randint(0, 100000) for x in range(repeats)]

# Trenowanie i zapisywanie wyników
results = {}
training_idx = 0
start_global_time = time.time()
for dataset_name, (train_loader, valid_loader) in datasets.items():
    for model_name, model in models[dataset_name].items():
        for config in configurations:
            data = {}
            for repeat, seed in enumerate(seeds):
                # Ustal losowy seed dla każdej powtórki
                print(f"\n=== Powtórka {repeat+1}/{repeats}, seed={seed} ===\n")
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                print(f'training_idx {training_idx}')
                print(f'Dataset_name: {dataset_name}')
                print(f'model_name: {model_name}')
                print(f'config: {config}')
                start_time = time.time()

                model = model.to(device)
                model_trained, history = train_es(
                    model, train_loader, valid_loader,
                    variant=config['variant'],
                    mu=config['mu'],
                    lambda_=config['lambda_'],
                    max_evals=15000,
                    device=device,
                    print_metrics=False,
                    seed=seed
                )
                elapsed_time = time.time() - start_time
                data[seed] = {'model': model_trained, 'history': history, 'training_time' : elapsed_time}

            key = f"{dataset_name}_{model_name}_{config['variant']}_mu{config['mu']}_lambda{config['lambda_']}"
            results[key] = data
            print(f"Czas trenowania: {elapsed_time:.2f} sekund")
            training_idx += 1

os.makedirs(os.path.dirname(output_path), exist_ok=True)

elapsed_global_time = time.time() - start_global_time
print(f'Eksperymenty trwały łacznie: {elapsed_global_time}')

def tensor_to_float(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    if isinstance(obj, dict):
        return {k: tensor_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_float(x) for x in obj]
    return obj

# Przygotowanie wyników do serializacji (model jako string z klasy i parametrami)
def model_to_str(model):
    return str(model)

results_serializable = {
    k: {
        seed : {
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

#results_serializable = {k: {'history': tensor_to_float(v['history']), 'training_time': tensor_to_float(v['training_time']), 'model': tensor_to_float(v['model'])} for k, v in results.items()}
