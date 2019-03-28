import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from time import time

data = pd.read_csv(Path("path/to/data"))


def augment_data(data):
    pass


class RandomForestRegressorExtended(RandomForestRegressor):
    def __init__(self, n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False):
        super().__init__(
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start)

    def train_model(self, train_data):
        start = time()  # track runtime
        self.fit(train_data[0], train_data[1])
        end = time()  # track runtime
        runtime = end - start
        print("Runtime: {:.4f}s", runtime)
