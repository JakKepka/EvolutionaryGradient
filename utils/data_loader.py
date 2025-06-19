import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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
