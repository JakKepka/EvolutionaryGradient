import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def load_mnist(batch_size=64, seed=42):
    """
    Ładuje zbiór danych MNIST, dzieli na train/val/test i zwraca DataLoadery oraz rozmiary.
    Każdy element wejściowy ma rozmiar (28*28,).
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # Transformacje: konwersja do tensora, normalizacja i spłaszczenie do (784,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Pobieranie zbioru treningowego
    full_train_dataset = datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )

    # Podział na train (60%) i val (20%) oraz test (20%)
    total_size = len(full_train_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    rest_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, _ = random_split(
        full_train_dataset, [train_size, val_size, rest_size], generator=generator
    )

    # Pobieranie zbioru testowego
    test_dataset = datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )

    # Tworzenie DataLoaderów
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = 28 * 28
    output_size = 10

    return train_loader, val_loader, test_loader, input_size, output_size

def load_wine_dataset(batch_size=32, seed=42):
    """
    Ładuje zbiór danych Wine, standaryzuje cechy, dzieli na train/val/test i zwraca DataLoadery oraz rozmiary.
    """
    # Wczytanie danych
    data = load_wine()
    X = data.data
    y = data.target

    # Standaryzacja cech
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Podział na zbiór train+val (60%) i test (40%) z zachowaniem rozkładu klas
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify=y, test_size=0.40, random_state=seed
    )

    # Podział train+val na train (60%) i val (20%) także stratyfikacja
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, stratify=y_train_val, test_size=0.5, random_state=seed
    )

    # Konwersja do tensorów PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Tworzenie datasetów i DataLoaderów
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    output_size = len(set(y))

    return train_loader, val_loader, test_loader, input_size, output_size

def load_breast_cancer_dataset(batch_size=32, seed=42):
    """
    Ładuje zbiór danych Breast Cancer Wisconsin, standaryzuje cechy, dzieli na train/val/test i zwraca DataLoadery oraz rozmiary.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify=y, test_size=0.40, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, stratify=y_train_val, test_size=0.5, random_state=seed
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    output_size = len(set(y))

    return train_loader, val_loader, test_loader, input_size, output_size

def load_iris_dataset(batch_size=32, seed=42):
    """
    Ładuje zbiór danych Iris, standaryzuje cechy, dzieli na train/val/test i zwraca DataLoadery oraz rozmiary.
    """
    data = load_iris()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify=y, test_size=0.40, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, stratify=y_train_val, test_size=0.5, random_state=seed
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    output_size = len(set(y))

    return train_loader, val_loader, test_loader, input_size, output_size

def load_steel_dataset(batch_size=32, seed=42):
    """
    Ładuje zbiór danych Steel, standaryzuje cechy, dzieli na train/val/test i zwraca DataLoadery oraz rozmiary.
    """
    import pandas as pd
    url = 'http://apmonitor.com/pds/uploads/Main/steel.txt'
    df = pd.read_csv(url)

    # Pierwsze 27 kolumn to cechy, ostatnie 7 to etykiety one-hot
    X = df.iloc[:, :-7].to_numpy()
    y_bin = df.iloc[:, -7:].to_numpy()
    y = np.argmax(y_bin, axis=1)

    # Standaryzacja cech
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Podział na train+val (60%) i test (40%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )

    # Podział train+val na train (50%) i val (50%) względem train_val (czyli 60/20 podział całości)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.5, stratify=y_train_val, random_state=seed
    )

    # Konwersja do tensorów PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Tworzenie DataLoaderów
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    output_size = len(np.unique(y))

    return train_loader, val_loader, test_loader, input_size, output_size


