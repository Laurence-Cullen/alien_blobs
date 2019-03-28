import numpy as np
from numba import jit


class Board:
    """
    nxn grid board
    """
    character_map = {}

    def __init__(self, board_size=9):
        self._board_size = board_size
        self._board = np.zeros((board_size, board_size, 2), dtype=np.int8)

    @staticmethod
    @jit(nopython=True)
    def legal_moves(board, board_size):
        moves_added = 0
        legal_moves_array = np.zeros(shape=(np.int8(board_size ** 2) - np.sum(board), 2), dtype=np.int8)

        for i in range(board_size):
            for j in range(board_size):
                if board[i][j][0] == 0 and board[i][j][1] == 0:
                    legal_moves_array[moves_added][0] = i
                    legal_moves_array[moves_added][1] = j
                    moves_added += 1

        return legal_moves_array

    # def legal_moves(self):
    #
    #     legal_moves_array = self.legal_moves_processing(
    #         board=self._board,
    #         board_size=self._board_size,
    #     )
    #
    #     return legal_moves_array

    def update_board(self, move, player_id):
        if self._board[move[0]][move[1]][0] == 0 and self._board[move[0]][move[1]][1] == 0:
            self._board[move[0]][move[1]][player_id] = 1
        else:
            raise ValueError(f"square:{move} already has a piece on it")

    def reset(self):
        self._board = np.zeros((self._board_size, self._board_size, 2))

    @property
    def board(self):
        return self._board

    @property
    def board_size(self):
        return self._board_size

    @staticmethod
    def square_to_char(square):
        """
        Map from a board square to the character to represent it.
        """
        if np.sum(square) == 0:
            return ' '
        elif np.sum(square) == 1:
            return str(np.argmax(square))
        else:
            raise ValueError(f'Square has invalid values:{square}')

    def __str__(self):
        """
        Return a string representation of the board in the following format:
        ------- i→
        |0|1| |
        -------
        | | |1|
        -------
        j
        ↓

        """

        rows = [
            '|' + '|'.join([Board.square_to_char(self._board[i][j][:]) for i in range(self._board_size)]) + '|\n'
            for
            j in range(self._board_size)
        ]

        divider = (2 * self._board_size + 1) * '-' + '\n'

        return divider + divider.join(rows) + divider


def main():
    board = Board()

    print(board.board)
    print(board.legal_moves())


if __name__ == '__main__':
    main()
