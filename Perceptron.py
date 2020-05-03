import numpy as np
import pandas as pd


class Perceptron(object):

    def __init__(self, eta=0.001, num_epochs=100):
        self.eta = eta
        self.num_epochs = num_epochs

    def fit(self, X, y):
        X = np.array(X)
        X = np.insert(X, 0, 1, 1)
        self.weights = np.random.normal(loc=0, scale=0.1, size=X.shape[1])
        for _ in range(self.num_epochs):
            for x_i, y_i in zip(X, y):
                error = (y_i - self.predict(x_i))
                self.weights += self.eta*error*x_i

    def net_input(self, x):
        return np.dot(x, self.weights)

    def threshold(self, z):
        return np.where(z >= 0, 1, -1)

    def predict(self, x):
        if x.ndim == 2:
            x = np.array(x)
            x = np.insert(x, 0, 1, 1)
        return self.threshold(self.net_input(x))
    
    def accuracy(self, X_test, y_test):
        return sum(self.predict(X_test) == y_test)/len(y_test)
    
    
if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split

    iris = pd.read_csv("iris.csv")

    X = iris.iloc[:, :4]
    X = X.iloc[:100]
    y = iris["class"].iloc[:100].apply(lambda c: 1 if c == "Iris-setosa" else -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    
    print("Test accuracy: {:.0%}".format(perceptron.accuracy(X_test, y_test)))
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
