import numpy as np
from sklearn.tree import DecisionTreeRegressor 

class GradientBoostRegression:
    def __init__(self, num_trees=100, learn_rate=0.1, tree_depth=3, l2reg=0.01):
        self.num_trees = num_trees
        self.learn_rate = learn_rate
        self.tree_depth = tree_depth
        self.l2reg = l2reg
        self.trees = []
        self.tree_weights = []
        self.start_pred = None

    def fit(self, x, y):
        self.start_pred = np.mean(y)
        current_pred = np.full(len(y), self.start_pred)

        for _ in range(self.num_trees):
            pred_err = y - current_pred
            
            tree = DecisionTreeRegressor(max_depth=self.tree_depth)
            tree.fit(x, pred_err)
            
            tree_pred = tree.predict(x)
            
            numerator = np.sum(pred_err * tree_pred)
            denominator = np.sum(tree_pred ** 2) + self.l2reg
            weight = numerator / denominator
            
            current_pred += self.learn_rate * weight * tree_pred
            
            self.trees.append(tree)
            self.tree_weights.append(weight)

    def predict(self, x):
        predictions = np.full(len(x), self.start_pred)
        for tree, weight in zip(self.trees, self.tree_weights):
            predictions += self.learn_rate * weight * tree.predict(x)
        return predictions

    def get_params(self, deep=True):
        return {
            'num_trees': self.num_trees,
            'learn_rate': self.learn_rate,
            'tree_depth': self.tree_depth,
            'reg_param': self.l2reg
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self