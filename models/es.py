import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import numpy as np

# class EvolutionStrategy:
#     def __init__(self, mu, rho, lambda_, fitness_func, generations):
#         self.mu = mu                # liczba rodziców
#         self.rho = rho              # liczba rodziców uczestniczących w rekombinacji
#         self.lambda_ = lambda_      # liczba potomków
#         self.fitness_func = fitness_func
#         self.generations = generations

#     def evolve(self, initial_weights):
#         n = initial_weights.shape[0]
#         tau_prime = 1 / np.sqrt(2 * n)
#         tau = 1 / np.sqrt(2 * np.sqrt(n))

#         # Inicjalizacja populacji: lista krotek (x, eta)
#         population = []
#         for _ in range(self.mu):
#             x = np.copy(initial_weights)
#             eta = np.random.uniform(0.1, 0.5, size=n)
#             population.append((x, eta))

#         for gen in range(self.generations):
#             offspring = []

#             for _ in range(self.lambda_):
#                 # Selekcja rodziców za pomocą turnieju binarnego
#                 parents = self._tournament_selection(population, self.rho)

#                 # Rekombinacja pośrednia
#                 x_recomb = np.mean([p[0] for p in parents], axis=0)
#                 eta_recomb = np.mean([p[1] for p in parents], axis=0)

#                 # Mutacja z adaptacją odchyleń
#                 N_global = np.random.randn()
#                 N_individual = np.random.randn(n)
#                 eta_mut = eta_recomb * np.exp(tau_prime * N_global + tau * N_individual)
#                 x_mut = x_recomb + eta_mut * np.random.randn(n)

#                 offspring.append((x_mut, eta_mut))

#             # Ocena fitness
#             combined_population = population + offspring
#             fitness_scores = [self.fitness_func(ind[0]) for ind in combined_population]

#             # Selekcja (μ + λ): wybór najlepszych μ osobników
#             sorted_indices = np.argsort(fitness_scores)[::-1]  # malejąco
#             population = [combined_population[i] for i in sorted_indices[:self.mu]]

#             # Opcjonalnie: wyświetlenie informacji o najlepszym fitnessie
#             best_fitness = fitness_scores[sorted_indices[0]]
#             print(f"Generacja {gen+1}: Najlepszy fitness = {best_fitness:.4f}")

#         # Zwrócenie wag najlepszego osobnika
#         best_individual = population[0][0]
#         return best_individual

#     def _tournament_selection(self, population, k):
#         selected = []
#         for _ in range(k):
#             i, j = np.random.choice(len(population), size=2, replace=False)
#             fit_i = self.fitness_func(population[i][0])
#             fit_j = self.fitness_func(population[j][0])
#             winner = population[i] if fit_i > fit_j else population[j]
#             selected.append(winner)
#         return selected

    
# def get_model_weights(model):
#     # Pobiera wagi modelu i konwertuje je do jednowymiarowego wektora NumPy
#     return np.concatenate([param.data.cpu().numpy().flatten() for param in model.parameters()])

# def set_model_weights(model, weights):
#     # Ustawia wagi modelu na podstawie jednowymiarowego wektora NumPy
#     pointer = 0
#     for param in model.parameters():
#         num_param = param.numel()
#         param.data = torch.from_numpy(weights[pointer:pointer + num_param].reshape(param.size())).to(param.device)
#         pointer += num_param

def evaluate(model, data_loader, device):
    # Ocena dokładności modelu na podanym zbiorze danych
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, dtype=torch.float64), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def train_es(
    model, train_loader, valid_loader=None,
    mu=30, rho=2, lamb=30, sigma=0.1,
    use_cauchy_variant=True,
    fitness_budget=15000, device='cpu'
):
    def get_weights():
        return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()
    
    def set_weights(w):
        pointer=0
        for p in model.parameters():
            num=p.numel()
            p.data.copy_(torch.from_numpy(w[pointer:pointer+num]).view_as(p))
            pointer+=num

    calls=0
    def fitness(w):
        nonlocal calls
        set_weights(w)
        loss=0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, dtype=torch.float64), targets.to(device)

                inputs, targets = inputs.to(device),targets.to(device)
                loss += F.cross_entropy(model(inputs), targets).item() * inputs.size(0)
        calls += 1
        return -loss/len(train_loader.dataset)

    dim = get_weights().shape[0]
    pop = [(get_weights(), sigma*np.ones(dim)) for _ in range(mu)]
    history=[]

    generation=0
    while calls < fitness_budget:
        generation+=1
        offspring=[]
        model.train()
        for _ in range(lamb):
            # tournament selection of ρ parents
            parents = []
            for _ in range(rho):
                a,b = np.random.choice(len(pop),2,replace=False)
                parents.append(pop[a] if fitness(pop[a][0]) > fitness(pop[b][0]) else pop[b])
            xs = np.mean([p[0] for p in parents], axis=0)
            es = np.mean([p[1] for p in parents], axis=0)

            if use_cauchy_variant and np.random.rand()<0.5:
                eps = np.random.standard_cauchy(dim)
            else:
                eps = np.random.randn(dim)

            xs_off = xs + es * eps
            es_off = es * np.exp((1/np.sqrt(2*dim))*np.random.randn() + (1/np.sqrt(2)*dim**0.25)*np.random.randn(dim))
            offspring.append((xs_off, es_off))

        combined = pop + offspring
        scores = [fitness(ind[0]) for ind in combined]

        # select best mu (µ+λ)
        idx = np.argsort(scores)[::-1][:mu]
        pop = [combined[i] for i in idx]

        best_w, best_e = pop[0]

        set_weights(best_w)
        # validation
        val_acc=None
        if valid_loader:
            correct=total=0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device, dtype=torch.float64), targets.to(device)

                    correct += (model(inputs).argmax(1)==targets).sum().item()
                    total += targets.size(0)
                val_acc = correct/total
        
        history.append((generation, scores[idx[0]], val_acc))
        print(f"Gen {generation}: fitness={scores[idx[0]]:.4f}, calls={calls}", end='')
        if val_acc is not None: print(f", val_acc={val_acc:.4f}")
        else: print()

        set_weights(pop[0][0])
    return model, history

