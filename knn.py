import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict classes for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use
        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """

        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)

        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X: np.array) -> np.array:
        """
        Computes L1 distance from every sample of X to every training sample
        Uses the simplest implementation with 2 Python loops
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sum(np.abs(X[i] - self.train_X[j]))

        return dists

    def compute_distances_one_loop(self, X: np.array) -> np.array:
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some calculations, so only 1 loop is used
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i] = np.sum(np.abs(X[i] - self.train_X), axis=1)

        return dists

    def compute_distances_no_loops(self, X: np.array) -> np.array:
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy
        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        return np.sum(np.abs(X[:, None, :] - self.train_X[None, :, :]), axis=2)

    def predict_labels_binary(self, distances: np.array) -> np.array:
        """
        Returns model predictions for binary classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        """

        # n_train = distances.shape[1]
        # n_test = distances.shape[0]
        # prediction = np.zeros(n_test)
        # pred = []

        # for val in range(n_test):
        # k_neighbours = distances.argsort()[val][:self.k]
        # nearest_neighbours = self.train_y[k_neighbours]
        # pred.append((nearest_neighbours == 0).sum() < (nearest_neighbours == 1).sum())
        # if (nearest_neighbours[val] == 0).sum() < (nearest_neighbours[val] == 1).sum():
        # prediction[val] = 1

        # n_train = distances.shape[1]
        # n_test = distances.shape[0]

        prediction = distances.argsort(axis=1)[:, :self.k]
        preds = []

        for i in prediction:
            if (self.train_y[i] == '0').sum() < (self.train_y[i] == '1').sum():
                preds.append(1)
            else:
                preds.append(0)

        return preds

    def predict_labels_multiclass(self, distances: np.array) -> np.array:
        """
        Returns model predictions for multi-class classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        """

        prediction = distances.argsort(axis=1)[:, :self.k]
        preds = []

        for i in prediction:
            preds.append(np.bincount(i).argmax())

        return np.array(self.train_y[preds])
