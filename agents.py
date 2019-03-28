import random
import numpy as np

class Player:
    def __init__(self, name, player_id=None):
        self._name = name
        self.player_id = player_id

    def next_move(self, board):
        pass

    @property
    def name(self):
        return self._name

    def set_id(self, player_id):
        self.player_id = player_id


class RandomPlayer(Player):
    """
    Plays on a random clear part of the board.
    """

    def next_move(self, board):
        return random.choice(board.legal_moves())


class ProximityRandomPlayer(Player):
    """
    Plays on a random part of the board that is adjacent to opponent pieces if possible.
    """

    def next_move(self, board):
        proximity_moves = board.legal_moves_adjacent_to_player(1 - self.player_id)

        if len(proximity_moves) > 0:
            return proximity_moves[np.random.choice(proximity_moves.shape[0])]
        else:
            legal_moves = board.legal_moves()
            return legal_moves[np.random.choice(legal_moves.shape[0])]
