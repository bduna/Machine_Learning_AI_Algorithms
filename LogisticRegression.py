import pandas as pd
import numpy as np

class LogisticRegression(object):

    def __init__(self, eta=0.01, num_epochs=20):
        self.eta = eta
        self.num_epochs = num_epochs
        self.weights = None
        self._cost = None

    def fit(self, X, y):
        y = np.array(y)
        self._cost = []
        X = np.array(X)
        X = np.insert(X, 0, 1, 1)
        self.weights = np.random.normal(loc=0, scale=0.01, size=X.shape[1])
        for _ in range(self.num_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            delta_weights = self.eta * X.transpose().dot(errors)
            self.weights += delta_weights
            epsilon = 1e-5
            self._cost.append(-y.dot(np.log(output + epsilon)) - (1 - y).dot(np.log(1 - output + epsilon)))

    def net_input(self, X):
        return np.dot(X, self.weights)

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, new_data):
        if type(new_data) is list:
            new_data.insert(0, 1)
            return np.where(self.activation(self.net_input(new_data)) >= 0.5, 1, 0)
        elif (type(new_data) is pd.DataFrame) or (type(new_data) is np.ndarray):
            new_data = np.array(new_data)
            new_data = np.insert(new_data, 0, 1, 1)
            predictions = np.where(self.activation(self.net_input(new_data)) >= 0.5, 1, 0)
            return list(predictions)
        else:
            print("Input must be one dimensional list, Numpy Array or Pandas DataFrame.")

    def accuracy(self, X_test, y_test):
        if (type(X_test) is pd.DataFrame) or (X_test is np.ndarray):
            X_test = np.array(X_test)
            return sum(logit_model.predict(X_test) == y_test)/len(y_test)
        else:
            print("Feature input must be Pandas DataFrame or Numpy ndarray.")

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    
    iris = pd.read_csv("iris.csv")

    logit_model = LogisticRegression(num_epochs=100, eta=0.01)
    X = iris[["sepal_length", "sepal_width"]]
    X = X.iloc[:100]
    y = iris["class"].iloc[:100].apply(lambda c: 1 if c == "Iris-setosa" else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    
    logit_model.fit(X_train, y_train)
    
    print(logit_model.weights)
    
    print("Logistic Regression model accuracy: {:.0%}".format(logit_model.accuracy(X_test, y_test)))
    
    



