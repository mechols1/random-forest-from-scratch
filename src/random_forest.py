import numpy as np
from src.decision_tree import DecisionTree
from multiprocessing import Pool


def _train_tree(args):
    X_sample, y_sample, mode = args
    tree = DecisionTree(mode=mode)
    tree.fit(X_sample, y_sample)
    return tree


class RandomForest():
    def __init__(self, mode, tree_count=10, feature_count=None):
        modes = ['classification', 'regression']
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}, got '{mode}'")
        self.mode = mode
        self.tree_count = tree_count
        self.feature_count = feature_count
        self.trained_trees = []

    def fit(self, X, y):
        if self.feature_count is None:
            self.feature_count = int(np.sqrt(X.shape[1]))

        n_samples = X.shape[0]
        samples = []
        for n in range(self.tree_count):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            samples.append((X[indices], y[indices], self.mode))

        with Pool() as pool:
            self.trained_trees = pool.map(_train_tree, samples)

    def predict(self, x):
        all_predictions = []
        predictions = []
        for tree in self.trained_trees:
            all_predictions.append(tree.predict(x))

        all_predictions = np.array(all_predictions)

        if self.mode == 'classification':
            def aggregate(col): return np.bincount(col).argmax()
        else:
            def aggregate(col): return np.mean(col)
        for i in range(all_predictions.shape[1]):
            column = all_predictions[:, i]
            predictions.append(aggregate(column))
        return predictions
