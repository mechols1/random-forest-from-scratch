import numpy as np
from node import Node


class DecisionTree:
    def __init__(self, max_depth=None, min_samples=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self._traverse(row, self.root))
        return predictions

    def _build_tree(self, X, y, depth=0):
        if (depth == self.max_depth or len(y) < self.min_samples or
                len(np.unique(y)) == 1):
            leaf = Node()
            leaf.predicted_class = np.bincount(y).argmax()
            return leaf
        feature_idx, value = self._best_split(X, y)
        left_mask = X[:, feature_idx] <= value
        right_mask = X[:, feature_idx] > value
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        left_subtree = self._build_tree(X_left, y_left, depth+1)
        right_subtree = self._build_tree(X_right, y_right, depth+1)
        internal_node = Node()
        internal_node.feature_idx = feature_idx
        internal_node.value = value
        internal_node.left = left_subtree
        internal_node.right = right_subtree
        return internal_node

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        for feature_idx in range(X.shape[1]):
            for threshold in np.unique(X[:, feature_idx]):
                # split left and right
                left_mask = X[:, feature_idx] <= threshold
                right_mask = X[:, feature_idx] > threshold
                y_left = y[left_mask]
                y_right = y[right_mask]
                weighted_gini = (len(y_left)/len(y) * self._gini(y_left) +
                                 len(y_right)/len(y) * self._gini(y_right))
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini(self, y):
        counts = np.bincount(y)
        proportions = counts / len(y)
        return 1 - (np.sum(np.square(proportions)))

    def _traverse(self, x, node):
        if node.left is None and node.right is None:
            return node.predicted_class
        if x[node.feature_idx] <= node.value:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
