import numpy as np


class Board():
    """
    9x9 grid board
    """

    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size, 2))

    def legal_moves(self):
        legal_moves = [[i, j] for i in range(9) for j in range(9) if
                       ((self.board[i][j][0] == 0) and (self.board[i][j][1] == 0))]
        return legal_moves

    def update_board(self, move, player_id):
        if move in self.legal_moves():
            self.board[move[0]][move[1]][player_id] = 1
        else:
            raise ValueError("{move} is illegal! Please enter moves in the format [i,j]")

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size, 2))
