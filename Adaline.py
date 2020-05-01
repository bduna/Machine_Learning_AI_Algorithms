import numpy as np


class Adaline:
    # Adaptive Linear Neurons (Adaline)

    def __init__(self, eta=0.01, num_epochs=10):
        self.eta = eta
        self.num_epochs = num_epochs
        self.weights = None
        self.cost_ = None

    def fit(self, X, y):
        self.cost_ = []
        X.insert(0, "bias", 1)
        self.weights = np.random.normal(loc=0, scale=0.01, size=X.shape[1])
        for _ in range(self.num_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            delta_weights = self.eta*X.T.dot(errors)
            self.weights += delta_weights
            self.cost_.append((errors**2).sum()/2)

    def net_input(self, X):
        return np.dot(X, self.weights)

    def activation(self, x):
        return x
    
    def predict(self, x):
        return 1 if self.activation(self.net_input(x)) >= 0 else -1

    
