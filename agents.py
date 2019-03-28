import random
import numpy as np
from numba import jit


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

    @staticmethod
    @jit(nopython=True)
    def legal_moves_adjacent_to_opponent(player_id, board, legal_moves):
        legal_moves_adjacent_to_opponent = np.zeros(shape=legal_moves.shape, dtype=np.int8)
        moves_added = 0

        for i in range(legal_moves.shape[0]):
            move = legal_moves[i][:]
            at_top = False
            at_left = False
            at_right = False
            at_bottom = False

            if move[0] == 0:
                at_left = True
            if move[0] == board.shape[0] - 1:
                at_right = True
            if move[1] == 0:
                at_top = True
            if move[1] == board.shape[1] - 1:
                at_bottom = True

            if not at_left:
                if board[move[0] - 1][move[1]][1 - player_id] == 1:
                    legal_moves_adjacent_to_opponent[moves_added] = move
                    moves_added += 1

            if not at_right:
                if board[move[0] + 1][move[1]][1 - player_id] == 1:
                    legal_moves_adjacent_to_opponent[moves_added] = move
                    moves_added += 1

            if not at_top:
                if board[move[0]][move[1] + 1][1 - player_id] == 1:
                    legal_moves_adjacent_to_opponent[moves_added] = move
                    moves_added += 1

            if not at_bottom:
                if board[move[0]][move[1] - 1][1 - player_id] == 1:
                    legal_moves_adjacent_to_opponent[moves_added] = move
                    moves_added += 1

        return legal_moves_adjacent_to_opponent[0:moves_added]

    def next_move(self, board):
        proximity_moves = self.legal_moves_adjacent_to_opponent(self.player_id, board.board, board.legal_moves())

        if len(proximity_moves) > 0:
            return random.choice(proximity_moves)
        else:
            return random.choice(board.legal_moves())
