import numpy as np
import pandas as pd


class Perceptron(object):

    def __init__(self, eta=0.001, num_epochs=50):
        self.eta = eta
        self.num_epochs = num_epochs

    def fit(self, X, y):
        self.w = np.random.normal(loc=0, scale=0.1, size=X.shape[1]+1)
        for _ in range(self.num_epochs):
            for x_i, y_i in zip(X.values, y):
                error = (y_i - self.predict(x_i))
                self.w[1:] += self.eta*error*x_i
                self.w[0] += self.eta*error
        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def threshold(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        return self.threshold(self.net_input(x))

    
if __name__ == "__main__":

    p = Perceptron()
    df = pd.read_csv("iris.csv", header=None)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    X = df.loc[0:100, ["sepal_length", "petal_length"]]
    y = df.loc[0:100, "class"]
    y = np.where(y == "Iris-setosa", 1, 0)
    p.fit(X, y)
