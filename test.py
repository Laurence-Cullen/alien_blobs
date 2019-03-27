import numpy as np


class Board():
    """
    9x9 grid board
    """

    def __init__(self, board_size=9):
        self.board = np.zeros((board_size, board_size))

    def legal_moves(self, move):
        legal_moves = [[i,j] for i in range(9) for j in range(9) if self.board[i][j] == 0]
        return legal_moves

NineByNine = Board()

NineByNine.legal_moves()
