import pandas as pd
import numpy as np

class LogisticRegression:

    def __init__(self, eta=0.01, num_epochs=20):
        self.eta = eta
        self.num_epochs = num_epochs
        self.weights = None
        self.cost_ = None

    def fit(self, X, y):
        y = np.array(y)
        self.cost_ = []
        X.insert(0, "bias", 1)
        self.weights = np.random.normal(loc=0, scale=0.01, size=X.shape[1])
        for _ in range(self.num_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            delta_weights = self.eta * X.T.dot(errors)
            self.weights += delta_weights
            epsilon = 1e-5
            self.cost_.append(-y.dot(np.log(output + epsilon)) - (1 - y).dot(np.log(1 - output + epsilon)))

    def net_input(self, X):
        return np.dot(X, self.weights)

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, new_data):
        if type(new_data) is list:
            new_data.insert(0, 1)
            return np.where(self.activation(self.net_input(new_data)) >= 0.5, 1, 0)
        elif type(new_data) is pd.DataFrame:
            predictions = []
            for index, x in new_data.iterrows():
                x = list(x)
                x.insert(0, 1)
                prediction = np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)
                predictions.append(prediction)
            return predictions
        else:
            print("Input must be one dimensional list or Pandas DataFrame.")

    def accuracy(self, X_test, y_test):
        num_trials = len(y_test)
        correct = 0
        if (type(X_test) is pd.DataFrame) and (type(y_test) == pd.core.series.Series):
            for index, x in X_test.iterrows():
                x = list(x)
                if self.predict(x) == y_test[index]:
                    correct += 1
            return correct / num_trials
        else:
            print("Input must be Pandas DataFrame and Series.")

if __name__ == "__main__":

    data = pd.read_csv("iris.csv")

    from sklearn.model_selection import train_test_split

    logit_model = LogisticRegression(num_epochs=100, eta=0.01)
    X = data[["sepal_length", "sepal_width"]]
    X = X.iloc[:100]
    y = data["class"].iloc[:100].apply(lambda c: 1 if c == "Iris-setosa" else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    logit_model.fit(X_train, y_train)

    print("Logistic regression weights:\n", logit_model.weights)
    print("Accuracy on test data: ", logit_model.accuracy(X_test, y_test))





