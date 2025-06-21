import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist(batch_size=64, shuffle=True, num_workers=0, valid_size=0.1, seed=42):
    """
    Ładuje zbiór danych MNIST i zwraca obiekty DataLoader dla zbiorów treningowego, walidacyjnego i testowego.

    Parametry:
    - batch_size (int): Rozmiar partii danych.
    - shuffle (bool): Czy losowo mieszać dane treningowe.
    - num_workers (int): Liczba procesów do ładowania danych.
    - valid_size (float): Proporcja zbioru walidacyjnego względem zbioru treningowego.
    - seed (int): Wartość ziarna dla generatora losowego (zapewnia powtarzalność podziału).

    Zwraca:
    - train_loader (DataLoader): DataLoader dla zbioru treningowego.
    - valid_loader (DataLoader): DataLoader dla zbioru walidacyjnego.
    - test_loader (DataLoader): DataLoader dla zbioru testowego.
    - input_size (int): Rozmiar wejścia (liczba cech).
    - output_size (int): Liczba klas wyjściowych.
    """

    # Transformacje: konwersja do tensora i normalizacja
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Pobieranie zbioru treningowego
    full_train_dataset = datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )

    # Obliczanie rozmiarów zbiorów treningowego i walidacyjnego
    total_train_size = len(full_train_dataset)
    valid_size = int(valid_size * total_train_size)
    train_size = total_train_size - valid_size

    # Ustawienie generatora losowego z określonym ziarnem
    generator = torch.Generator().manual_seed(seed)

    # Podział zbioru na treningowy i walidacyjny
    train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size], generator=generator)

    # Pobieranie zbioru testowego
    test_dataset = datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )

    # Tworzenie DataLoaderów
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Rozmiar wejścia: 28x28 pikseli = 784 cech
    input_size = 28 * 28

    # Liczba klas: cyfry od 0 do 9
    output_size = 10

    return train_loader, valid_loader, test_loader, input_size, output_size


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