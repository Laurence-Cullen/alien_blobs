from agents import Player
import pickle
from copy import deepcopy
import numpy as np


# load the model from disk
#forest_regressor = pickle.load(open('random_forest_model.sav', 'rb'))


class RandomForestPlayer(Player):
    def __init__(self, name, forest_regressor, player_id=None):
        super().__init__(name, player_id)
        self.forest_regressor = forest_regressor

    def board_value_assessment(self, board):
        future_board_values = []
        legal_moves = board.legal_moves(board.board, board.board_size)
        for move in legal_moves:
            temp_board = deepcopy(board)
            temp_board.update_board(move, self.player_id)
            future_board_values.append(
                [move, self.forest_regressor.predict(np.reshape(temp_board.board, (1, board.board_size ** 2 * 2)))])

        return future_board_values

    def next_move(self, board):
        future_board_values = self.board_value_assessment(board)
        if self.player_id == 0:
            return max(future_board_values, key=lambda x: x[1])[0]

        if self.player_id == 1:
            return min(future_board_values, key=lambda x: x[1])[0]
