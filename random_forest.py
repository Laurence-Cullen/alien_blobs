import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import pickle


class RandomForestRegressorExtended(RandomForestRegressor):
    def __init__(self, n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False):
        super().__init__(
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs,
            random_state, verbose, warm_start)
        
        self.train_data = None
        self.train_targets = None

    def load_data(self, folder_path, board_size):
        folder_path = Path(folder_path)
        draw_board_paths = folder_path.glob('-1/*.npy')
        player_one_win_board_paths = folder_path.glob('0/*.npy')
        player_two_win_board_paths = folder_path.glob('1/*.npy')

        board_count = len(list(draw_board_paths)) + len(list(player_one_win_board_paths)) + len(list(player_two_win_board_paths))
        board_flattened_length = int(board_size ** 2)*2
        self.train_data = np.zeros((board_count, board_flattened_length))
        self.train_targets = np.zeros((board_count, 1))
        row = 0

        for draw_board_path in draw_board_paths:
            self.train_data[row][:] = np.load(draw_board_path).flatten()
            self.train_targets[row] = 0.5
            row += 1

        for player_one_win_board_path in player_one_win_board_paths:
            self.train_data[row][:] = np.load(player_one_win_board_path).flatten()
            self.train_targets[row] = 0
            row += 1

        for player_two_win_board_path in player_two_win_board_paths:
            self.train_data[row][:] = np.load(player_two_win_board_path).flatten()
            self.train_targets[row] = 1
            row += 1

    def skfold_fit_predict(self):
        # create skfold
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_scores = []
        start = time()  # track runtime

        for i, (train_index, test_index) in enumerate(skfold.split(self.train_data, self.train_targets)):
            X_train, X_valid = self.train_data.iloc[train_index], self.train_data.iloc[test_index]
            y_train, y_valid = self.train_targets.iloc[train_index], self.train_targets.iloc[test_index]

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


forest_regressor = RandomForestRegressorExtended()
forest_regressor.load_data('data', 9)
#forest_regressor.skfold_fit_predict()
print(forest_regressor.train_targets)
print(forest_regressor.train_data)