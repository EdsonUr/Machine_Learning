import numpy as np

class Node():
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, information_gain=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree():

    def __init__(self, max_depth=10, min_samples_split=2, min_information_gain=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.root = None
        self.n_features = None

    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def _information_gain(self, y, left_indices, right_indices, parent_entropy):
        m = len(y)
        
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        
        w_left = len(left_indices) / m
        w_right = len(right_indices) / m

        child_entropy = (w_left * left_entropy + w_right * right_entropy)

        gain = parent_entropy - child_entropy
        return gain
    
    def _best_split(self, X, y):
        best_gain = -np.inf
        best_split_info = {}
        parent_entropy = self._entropy(y)

        for feature_index in range(X.shape[1]):
            unique_values = np.sort(np.unique(X[:, feature_index]))
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gain = self._information_gain(y, left_indices, right_indices, parent_entropy)

                if gain > best_gain:
                    best_gain = gain
                    best_split_info = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'information_gain': gain
                    }
        
        return best_split_info
    
    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        best = self._best_split(X, y)

        if not best or best['information_gain'] < self.min_information_gain:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        left_subtree = self._build_tree(X[best['left_indices']], y[best['left_indices']], depth + 1)
        right_subtree = self._build_tree(X[best['right_indices']], y[best['right_indices']], depth + 1)

        return Node(
            feature_index = best['feature_index'],
            threshold = best['threshold'],
            left = left_subtree,
            right = right_subtree,
            information_gain = best['information_gain']
        )

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        return self
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    