import numpy as np

class EvolutionStrategy:
    def __init__(self, pop_size, sigma, learning_rate, fitness_func):
        self.pop_size = pop_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.fitness_func = fitness_func

    def evolve(self, weights, generations):
        for gen in range(generations):
            noise = np.random.randn(self.pop_size, *weights.shape)
            rewards = np.array([self.fitness_func(weights + self.sigma * n) for n in noise])
            normalized_rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            weights += self.learning_rate / (self.pop_size * self.sigma) * np.dot(noise.T, normalized_rewards).T
        return weights
