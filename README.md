# EvolutionaryGradient

The goal of this project is to compare the effectiveness of selected evolutionary methods and a gradient-based method in the context of neural network training. The project will involve the implementation and evaluation of the following approaches:

- **A variant of Differential Evolution (DE)** – the DEAW algorithm from publication [2], which introduces adaptive weight boundaries to the classic DE to improve the efficiency of neural network training.
- **A variant of Evolution Strategy (ES)** – inspired by [1], with a selected modification of the classic evolution strategy.
- **Gradient-based method** – the Adam optimizer, as a representative method based on backpropagation.

The specific variants of differential evolution and evolution strategy are still subject to change, as we aim to create a strong alternative to gradient-based training.

The evolutionary methods will operate in the weight space of the neural network, without using loss function derivatives. The aim is to compare the effectiveness of each method in terms of:

- Classification accuracy on training and test datasets,
- Convergence speed (number of iterations and training time),
- Resistance to local minima,
- Impact of parameters (e.g., population size, mutation rate, etc.).

This comparative study will help evaluate the practical viability of evolutionary algorithms as alternatives to gradient-based optimization in neural network training.
