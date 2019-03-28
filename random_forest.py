import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import pickle

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

    def skfold_fit_predict(self, train_data, train_targets):
        # create skfold
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_scores = []
        start = time()  # track runtime

        for i, (train_index, test_index) in enumerate(skfold.split(train_data, train_targets)):
            X_train, X_valid = train_data.iloc[train_index], train_data.iloc[test_index]
            y_train, y_valid = train_targets.iloc[train_index], train_targets.iloc[test_index]

            self.fit(X_train, y_train)  # fit model

            # Score evaluation with given metric 
            mean_squared_error_score = mean_squared_error(y_valid, self.predict(X_valid))
            fold_scores.append(mean_squared_error_score)

        end = time()  # track runtime
        runtime = end - start

        # save the model to disk
        filename = 'random_forest_model.sav'
        pickle.dump(self, open(filename, 'wb'))

        print("Model fold scores: {}".format(fold_scores))
        return print("Model average mean squared error score: {:.4f}   Runtime: {:.2f}s \n".format(
                                                                                                      np.mean(
                                                                                                          fold_scores),
                                                                                                      runtime))

