import torch
from torchvision import datasets, transforms

def load_mnist(batch_size=64, shuffle=True, num_workers=2):
    """
    Ładuje zbiór danych MNIST i zwraca obiekty DataLoader dla zbiorów treningowego i testowego.

    Parametry:
    - batch_size (int): Rozmiar partii danych.
    - shuffle (bool): Czy losowo mieszać dane treningowe.
    - num_workers (int): Liczba procesów do ładowania danych.

    Zwraca:
    - train_loader (DataLoader): DataLoader dla zbioru treningowego.
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
    train_dataset = datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )

    # Pobieranie zbioru testowego
    test_dataset = datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )

    # Tworzenie DataLoaderów
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Rozmiar wejścia: 28x28 pikseli = 784 cech
    input_size = 28 * 28

    # Liczba klas: cyfry od 0 do 9
    output_size = 10

    return train_loader, test_loader, input_size, output_size
