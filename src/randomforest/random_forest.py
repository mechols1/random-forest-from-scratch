import numpy as np
from decision_tree import DecisionTree


class RandomForest():
    def __init__(self, tree_count=10, feature_count=None):
        self.tree_count = tree_count
        self.feature_count = feature_count
        self.trained_trees = []

    def fit(self, X, y):
        if self.feature_count is None:
            self.feature_count = int(np.sqrt(X.shape[1]))

        n_samples = X.shape[0]
        for n in range(self.tree_count):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree()
            tree.fit(X_sample, y_sample)
            self.trained_trees.append(tree)

    def predict(self, X):
        all_predictions = []
        votes = []
        for tree in self.trained_trees:
            all_predictions.append(tree.predict(X))

        all_predictions = np.array(all_predictions)

        for i in range(all_predictions.shape[1]):
            column = all_predictions[:, i]
            majority_vote = np.bincount(column).argmax()
            votes.append(majority_vote)
        return votes
