import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Cauchy

# Evaluate an individual's fitness (total cross-entropy loss)
def evaluate_individual(model, weights, train_loader, device):
    model.set_weights(weights)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
    return total_loss

# Evaluate model on a dataset and return average loss and accuracy
def evaluate_model(model, weights, data_loader, device):
    model.set_weights(weights)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Binary tournament selection
def binary_tournament_selection(fitnesses, k):
    selected = []
    for _ in range(k):
        idx1, idx2 = np.random.choice(len(fitnesses), 2, replace=False)
        if fitnesses[idx1] < fitnesses[idx2]:
            selected.append(idx1)
        else:
            selected.append(idx2)
    return selected

# Main training function with training history
def train_es(model, train_loader, valid_loader, variant_mu_lambda=True, modified_ES=True, mu=30, lambda_=30, max_evals=2000, device='cpu', print_metrics=True, seed=42):
    # Initialize model
    n_weights = model.get_weights().numel()

    # Mutation parameters
    tau_prime = 1 / np.sqrt(2 * n_weights)
    tau = 1 / np.sqrt(2 * np.sqrt(n_weights))

    # Initialize population
    population_weights = [model.get_weights().clone() for _ in range(mu)]
    population_eta = [torch.rand(n_weights, device=device) for _ in range(mu)]
    fitnesses = [evaluate_individual(model, weights, train_loader, device) for weights in population_weights]
    function_evals = mu
    g = 0

    # Initialize training history
    history = []

    while function_evals < max_evals:
        offspring_weights = []
        offspring_eta = []
        for _ in range(lambda_):
            if modified_ES != True:
                # Select 2 parents
                parent_indices = binary_tournament_selection(fitnesses, 2)
                parent1_weights = population_weights[parent_indices[0]]
                parent2_weights = population_weights[parent_indices[1]]
                parent1_eta = population_eta[parent_indices[0]]
                parent2_eta = population_eta[parent_indices[1]]
                
                # Intermediate recombination
                r = torch.rand(1, device=device)
                child_weights = r * parent1_weights + (1 - r) * parent2_weights
                child_eta = r * parent1_eta + (1 - r) * parent2_eta
                
                # Normal mutation
                global_noise = torch.randn(1, device=device)
                component_noise = torch.randn(n_weights, device=device)
                eta_new = child_eta * torch.exp(tau_prime * global_noise + tau * component_noise)
                child_weights = child_weights + eta_new * torch.randn(n_weights, device=device)
                child_eta = eta_new
            else:
                # Select 1 parent
                parent_index = binary_tournament_selection(fitnesses, 1)[0]
                parent_weights = population_weights[parent_index]
                parent_eta = population_eta[parent_index]
                r = torch.rand(1, device=device)
                if r < 0.5:
                    # Normal mutation
                    global_noise = torch.randn(1, device=device)
                    component_noise = torch.randn(n_weights, device=device)
                    eta_new = parent_eta * torch.exp(tau_prime * global_noise + tau * component_noise)
                    child_weights = parent_weights + eta_new * torch.randn(n_weights, device=device)
                    child_eta = eta_new
                else:
                    # Cauchy mutation
                    cauchy_dist = Cauchy(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
                    delta = cauchy_dist.sample([n_weights]).squeeze()
                    child_weights = parent_weights + parent_eta * delta
                    child_eta = parent_eta
            
            # Clamp weights
            child_weights = torch.clamp(child_weights, -1000, 1000)
            offspring_weights.append(child_weights)
            offspring_eta.append(child_eta)
        
        # Evaluate offspring
        offspring_fitnesses = [evaluate_individual(model, weights, train_loader, device) for weights in offspring_weights]
        function_evals += lambda_
        
        # Selection
        if variant_mu_lambda == True:
            indices = np.argsort(offspring_fitnesses)[:mu]
            population_weights = [offspring_weights[i] for i in indices]
            population_eta = [offspring_eta[i] for i in indices]
            fitnesses = [offspring_fitnesses[i] for i in indices]
        else:
            all_weights = population_weights + offspring_weights
            all_eta = population_eta + offspring_eta
            all_fitnesses = fitnesses + offspring_fitnesses
            indices = np.argsort(all_fitnesses)[:mu]
            population_weights = [all_weights[i] for i in indices]
            population_eta = [all_eta[i] for i in indices]
            fitnesses = [all_fitnesses[i] for i in indices]
        
        g += 1
        # Find best individual
        best_index = np.argmin(fitnesses)
        best_weights = population_weights[best_index]
        
        # Evaluate on training set
        train_loss, train_acc = evaluate_model(model, best_weights, train_loader, device)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, best_weights, valid_loader, device)
        
        # Store statistics in history
        history.append({
            'generation': g,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'best_fitness': min(fitnesses)
        })
        if print_metrics == True:
            # Print results for each generation
            print(f"Generation {g}:")
            print(f"  Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            print(f"  Best Fitness: {min(fitnesses):.4f}")
    
    # Set best weights
    best_index = np.argmin(fitnesses)
    model.set_weights(population_weights[best_index])
    
    # Final validation
    val_loss, val_acc = evaluate_model(model, population_weights[best_index], valid_loader, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return model, history