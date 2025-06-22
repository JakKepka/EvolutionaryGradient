import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from scipy.stats import cauchy
from torch.distributions import Cauchy

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute accuracy for evaluation
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Fitness function using cross-entropy loss
def fitness(weights, model, train_loader, device):
    model.set_weights(weights.to(device))
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(train_loader.dataset)

# Evaluate model with given weights (optional)
def evaluate_model(model, data_loader, device, weights=None):
    model.eval()
    if weights is not None:
        # Ensure weights is a tensor and move it to the device
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        model.set_weights(weights.to(device))
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# DE Algorithm
def train_de(model, train_loader, valid_loader, device, NP=50, F=0.5, CR=0.9, max_generations=100, initial_lower=-1.0, initial_upper=1.0):
    num_weights = sum(p.numel() for p in model.parameters())
    population = np.random.uniform(initial_lower, initial_upper, (NP, num_weights))
    fitnesses = np.array([fitness(torch.tensor(p, dtype=torch.float32), model, train_loader, device) for p in population])

    history = []
    for generation in range(max_generations):
        for i in range(NP):
            candidates = [j for j in range(NP) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            v = population[a] + F * (population[b] - population[c])

            # Stałe ograniczenie: ograniczamy wartości do zakresu
            v = np.clip(v, initial_lower, initial_upper)

            u = np.copy(population[i])
            j_rand = np.random.randint(0, num_weights)
            for j in range(num_weights):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            u_tensor = torch.tensor(u, dtype=torch.float32)
            loss_u = fitness(u_tensor, model, train_loader, device)

            if loss_u < fitnesses[i]:
                population[i] = u.copy()
                fitnesses[i] = loss_u

        # Evaluate best individual of the generation
        best_idx = np.argmin(fitnesses)
        best_weights = torch.tensor(population[best_idx], dtype=torch.float32)
        train_loss, train_acc = evaluate_model(model, train_loader, device, best_weights)
        val_loss, val_acc = evaluate_model(model, valid_loader, device, best_weights)
        history.append({
            'generation': generation + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'best_fitness': fitnesses[best_idx]
        })

    # Set best weights and return
    best_idx = np.argmin(fitnesses)
    best_weights = torch.tensor(population[best_idx], dtype=torch.float32)
    model.set_weights(best_weights.to(device))
    return model, history

# DEAW Algorithm
def train_deaw(model, train_loader, valid_loader, device, NP=50, F=0.5, CR=0.9, max_generations=100, initial_lower=-1.0, initial_upper=1.0):
    num_weights = sum(p.numel() for p in model.parameters())
    lower_bounds = np.full(num_weights, initial_lower)
    upper_bounds = np.full(num_weights, initial_upper)
    population = np.random.uniform(initial_lower, initial_upper, (NP, num_weights))
    fitnesses = np.array([fitness(torch.tensor(p, dtype=torch.float32), model, train_loader, device) for p in population])

    history = []
    for generation in range(max_generations):
        for i in range(NP):
            candidates = [j for j in range(NP) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            v = population[a] + F * (population[b] - population[c])

            for j in range(num_weights):
                if v[j] < lower_bounds[j]:
                    lower_bounds[j] *= 3
                    v[j] = lower_bounds[j]
                elif v[j] > upper_bounds[j]:
                    upper_bounds[j] *= 3
                    v[j] = upper_bounds[j]

            u = np.copy(population[i])
            j_rand = np.random.randint(0, num_weights)
            for j in range(num_weights):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            u_tensor = torch.tensor(u, dtype=torch.float32)
            loss_u = fitness(u_tensor, model, train_loader, device)

            if loss_u < fitnesses[i]:
                population[i] = u.copy()
                fitnesses[i] = loss_u

        # Evaluate best individual of the generation
        best_idx = np.argmin(fitnesses)
        best_weights = torch.tensor(population[best_idx], dtype=torch.float32)
        train_loss, train_acc = evaluate_model(model, train_loader, device, best_weights)
        val_loss, val_acc = evaluate_model(model, valid_loader, device, best_weights)
        history.append({
            'generation': generation + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'best_fitness': fitnesses[best_idx]
        })

    # Set best weights and return
    best_idx = np.argmin(fitnesses)
    best_weights = torch.tensor(population[best_idx], dtype=torch.float32)
    model.set_weights(best_weights.to(device))
    return model, history

# Compute cross-entropy loss for a batch
def compute_loss(model, inputs, targets):
    outputs = model(inputs)
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(outputs, targets).item()


# Population-based Adam (P-Adam) with batch processing
class PAdam:
    def __init__(self, population, alpha=0.1, gamma1=0.9, gamma2=0.99, gamma3=0.999, tau=1e-7):
        self.population = population
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.tau = tau
        self.m = [torch.zeros_like(ind) for ind in population]  # First moment
        self.n = [torch.zeros_like(ind) for ind in population]  # Second moment

    def step(self, model, data_loader, t):
        fitnesses = []
        loss_fn = nn.CrossEntropyLoss()

        for inputs, targets in data_loader:
            batch_fitnesses = []
            for i, (ind, m_i, n_i) in enumerate(zip(self.population, self.m, self.n)):
                model.zero_grad()  # Clear previous gradients
                model.set_weights(ind)  # Set model weights to current individual
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()  # Compute gradients for all parameters

                # Collect gradients for all parameters and flatten them
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

                # Update moments
                m_i = self.gamma1 * m_i + (1 - self.gamma1) * grads
                n_i = self.gamma2 * n_i + (1 - self.gamma3) * (grads ** 2)

                # Bias correction
                m_hat = m_i / (1 - self.gamma1 ** t)
                n_hat = n_i / (1 - self.gamma3 ** t)

                # Update parameters
                new_ind = ind - self.alpha * m_hat / (torch.sqrt(n_hat) + self.tau)
                self.population[i] = new_ind
                batch_fitnesses.append(loss.item())

                # Zero gradients again to avoid accumulation
                model.zero_grad()

                self.m[i] = m_i
                self.n[i] = n_i

            fitnesses.append(batch_fitnesses)

        # Average fitness across batches
        fitnesses = np.mean(fitnesses, axis=0).tolist()
        return fitnesses

# Modified CoBiDE (M-CoBiDE) with batch processing
class MCoBiDE:
    def __init__(self, population, pb=0.5, ps=0.4):
        self.population = population
        self.pb = pb
        self.ps = ps
        self.rng = np.random.default_rng()
        self.F = [self._sample_F() for _ in population]
        self.CR = [self._sample_CR() for _ in population]

    def _sample_F(self):
        r = self.rng.random()
        if r < 0.5:
            return cauchy.rvs(loc=0.65, scale=0.1, random_state=self.rng)
        else:
            return cauchy.rvs(loc=1.0, scale=0.1, random_state=self.rng)

    def _sample_CR(self):
        r = self.rng.random()
        if r < 0.5:
            cr = cauchy.rvs(loc=0.1, scale=0.1, random_state=self.rng)
        else:
            cr = cauchy.rvs(loc=0.95, scale=0.1, random_state=self.rng)
        return np.clip(cr, 0, 1)

    def step(self, model, data_loader):
        fitnesses = []
        for inputs, targets in data_loader:
            batch_fitnesses = [compute_loss(model, inputs, targets) for ind in self.population]
            best_idx = np.argmin(batch_fitnesses)
            new_population = []

            # Compute covariance matrix for top ps proportion
            top_indices = np.argsort(batch_fitnesses)[:int(self.ps * len(self.population))]
            top_pop = torch.stack([self.population[i] for i in top_indices])
            cov = torch.cov(top_pop.T)
            cov += 1e-6 * torch.eye(cov.shape[0])  # Add perturbation for stability
            eigvals, eigvecs = torch.linalg.eigh(cov)
            P = eigvecs

            for i, (ind, F_i, CR_i) in enumerate(zip(self.population, self.F, self.CR)):
                r1, r2 = self.rng.choice([j for j in range(len(self.population)) if j != i], 2, replace=False)
                v_i = ind + F_i * (self.population[best_idx] - ind) + F_i * (self.population[r1] - self.population[r2])

                r3 = self.rng.random()
                if r3 >= self.pb:
                    u_i = ind.clone()
                    j_rand = self.rng.integers(0, len(ind))
                    for j in range(len(ind)):
                        if self.rng.random() <= CR_i or j == j_rand:
                            u_i[j] = v_i[j]
                else:
                    x_prime = P.T @ ind
                    v_prime = P.T @ v_i
                    u_prime = x_prime.clone()
                    j_rand = self.rng.integers(0, len(ind))
                    for j in range(len(ind)):
                        if self.rng.random() <= CR_i or j == j_rand:
                            u_prime[j] = v_prime[j]
                    u_i = P @ u_prime

                model.set_weights(u_i)
                u_fitness = compute_loss(model, inputs, targets)
                if u_fitness < batch_fitnesses[i]:
                    new_population.append(u_i)
                    self.F[i] = self._sample_F()
                    self.CR[i] = self._sample_CR()
                else:
                    new_population.append(ind)

            self.population = new_population
            fitnesses.append(batch_fitnesses)

        # Average fitness across batches
        fitnesses = np.mean(fitnesses, axis=0).tolist()
        return fitnesses

# EDEAdam Algorithm with batch processing
class EDEAdam:
    def __init__(self, model, pop_size=50, max_evals=25000, exchange_interval=5, batch_size=32):
        self.model = model
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.exchange_interval = exchange_interval
        self.batch_size = batch_size
        self.dim = sum(p.numel() for p in model.parameters())  # Actual dimension from model parameters
        # Compute expected dimension based on architecture
        if hasattr(model, 'hidden_sizes'):  # Check if model is DeepMLP
            expected_dim = model.input_size * model.hidden_sizes[0] + model.hidden_sizes[0]  # Input to first hidden layer + bias
            for i in range(len(model.hidden_sizes) - 1):
                expected_dim += model.hidden_sizes[i] * model.hidden_sizes[i + 1] + model.hidden_sizes[i + 1]  # Hidden to hidden + bias
            expected_dim += model.hidden_sizes[-1] * model.output_size + model.output_size  # Last hidden to output + bias
        else:  # Assume model is MLP with hidden_size
            expected_dim = model.input_size * model.hidden_size + model.hidden_size + model.hidden_size * model.output_size + model.output_size
        assert self.dim == expected_dim, f"Dimension mismatch: got {self.dim}, expected {expected_dim}"
        # Initialize population
        self.population = [torch.rand(self.dim, device=next(model.parameters()).device) * 2 - 1 for _ in range(pop_size)]
        self.sub_pop1 = self.population[:pop_size//2]
        self.sub_pop2 = self.population[pop_size//2:]
        self.p_adam = PAdam(self.sub_pop1)
        self.m_cobide = MCoBiDE(self.sub_pop2)
        self.history = []

    def run(self, train_loader, valid_loader, device):
        t = 1
        eval_count = 0
        best_fitness = float('inf')
        best_individual = None

        while eval_count < self.max_evals:
            fitness1 = self.p_adam.step(self.model, train_loader, t)
            fitness2 = self.m_cobide.step(self.model, train_loader)
            eval_count += len(self.sub_pop1) + len(self.sub_pop2)

            best_idx1, worst_idx1 = np.argmin(fitness1), np.argmax(fitness1)
            best_idx2, worst_idx2 = np.argmin(fitness2), np.argmax(fitness2)

            if min(fitness1 + fitness2) < best_fitness:
                best_fitness = min(fitness1 + fitness2)
                best_individual = self.sub_pop1[best_idx1] if fitness1[best_idx1] < fitness2[best_idx2] else self.sub_pop2[best_idx2]

            if t % self.exchange_interval == 0:
                if fitness1[best_idx1] < fitness2[worst_idx2]:
                    self.sub_pop2[worst_idx2] = self.sub_pop1[best_idx1].clone()
                if fitness2[best_idx2] < fitness1[worst_idx1]:
                    self.sub_pop1[worst_idx1] = self.sub_pop2[best_idx2].clone()

            # Evaluate best individual of the iteration
            self.model.set_weights(best_individual.to(device))
            train_loss, train_acc = evaluate_model(self.model, train_loader, device, best_individual)
            val_loss, val_acc = evaluate_model(self.model, valid_loader, device, best_individual)
            self.history.append({
                'iteration': t,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'best_fitness': best_fitness
            })

            t += 1

        self.model.set_weights(best_individual.to(device))
        return self.model, self.history