import numpy as np

class DifferentialEvolution:
    def __init__(self, pop_size, mutation_factor, crossover_rate, bounds, fitness_func):
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.dim = len(bounds)
        self.population = self._initialize_population()

    def _initialize_population(self):
        return np.random.rand(self.pop_size, self.dim) * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]

    def evolve(self, generations):
        for gen in range(generations):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[:,0], self.bounds[:,1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                if self.fitness_func(trial) < self.fitness_func(self.population[i]):
                    self.population[i] = trial
