
import torch
import torch.nn.functional as F
import numpy as np
import copy

def train_es(train_loader, valid_loader, model_class,
             mu=30, lam=30, generations=50, seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Tworzymy tymczasowy model by poznać rozmiar wektora wag
    dummy_model = model_class()
    w_template = dummy_model.get_weights()
    n_params = w_template.numel()

    tau = 1 / np.sqrt(2 * np.sqrt(n_params))
    tau_prime = 1 / np.sqrt(2 * n_params)

    # Inicjalizacja populacji: lista (wagi, eta)
    def initialize_population():
        pop = []
        for _ in range(mu):
            x = np.random.randn(n_params).astype(np.float32)
            eta = np.random.rand(n_params).astype(np.float32)
            pop.append((x, eta))
        return pop

    # Mutacje
    def mutation_normal(x, eta):
        new_eta = eta * np.exp(tau_prime * np.random.randn() + tau * np.random.randn(*eta.shape))
        new_x = x + new_eta * np.random.randn(*x.shape)
        return new_x.astype(np.float32), new_eta.astype(np.float32)

    def mutation_cauchy(x, eta):
        new_eta = eta * np.exp(tau_prime * np.random.randn() + tau * np.random.randn(*eta.shape))
        new_x = x + eta * np.random.standard_cauchy(size=x.shape).astype(np.float32)
        return new_x.astype(np.float32), new_eta.astype(np.float32)

    # Ocena osobnika
    def evaluate(ind):
        weights = torch.tensor(ind[0])
        model = model_class()
        model.set_weights(weights)
        model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, reduction='sum')
                total_loss += loss.item()
                total_samples += yb.size(0)
        return total_loss / total_samples

    # Ocena accuracy
    def compute_accuracy(model, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                preds = torch.argmax(model(xb), dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

    # Główna pętla ES
    pop = initialize_population()
    history = []

    for gen in range(generations):
        offspring = []
        for _ in range(lam):
            idx = np.random.randint(0, mu)
            x, eta = pop[idx]
            if np.random.rand() < 0.5:
                new_x, new_eta = mutation_normal(x, eta)
            else:
                new_x, new_eta = mutation_cauchy(x, eta)
            offspring.append((new_x, new_eta))

        # Ocena i selekcja
        pop_all = pop + offspring
        pop_all.sort(key=evaluate)
        pop = pop_all[:mu]

        # Ewaluacja najlepszego
        best_weights = torch.tensor(pop[0][0])
        best_model = model_class()
        best_model.set_weights(best_weights)
        acc = compute_accuracy(best_model, valid_loader)
        history.append(acc)
        print(f"Generation {gen+1}: Validation accuracy = {acc:.4f}")

    return best_model, history
