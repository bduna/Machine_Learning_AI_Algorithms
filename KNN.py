import numpy as np
from collections import Counter

class KNN(object):
    
    def __init__(self, X=None, y=None, K = None):
        self.X = np.array(X)
        self.y = y
        self.K = K
        if (self.K is None) and (self.X is not None):
            self.K = int(np.floor(np.sqrt(self.X.shape[0])))
            
    def euclidean(self, x1, x2):
        return np.sqrt(np.dot(x1 - x2, x1 - x2))

    def load_data(self, data):
        self.X = np.array(data)
        if self.K is None:
            self.K = int(np.floor(np.sqrt(self.X.shape[0])))

    def _predict_instance(self, x):
        distances = []
        for index, row in enumerate(self.X):
            distances.append((index, self.euclidean(x, row)))
        votes = [self.y[t[0]] for t in sorted(distances, key=lambda x: x[1])[:self.K]]
        vote_cntr = Counter(votes)
        return max(vote_cntr)
            
    def predict(self, new_data):
        new_data = np.array(new_data)
        if new_data.ndim == 1:
            return self._predict_instance(new_data)
        else:
            predictions = []
            for x in new_data:
                predictions.append(self._predict_instance(x))
            return predictions
            
            
if __name__ == "__main__":

    import pandas as pd

    iris = pd.read_csv("iris.csv")

    X = iris.iloc[:, :4]
    y = iris["class"]

    knn = KNN(X=X, y=y)
    print("Automatically calculated K: ", knn.K)
    accuracy = sum(knn.predict(X) == y) / len(y)
    print("Accuracy on entire dataset: {:.0%}".format(accuracy))
    
