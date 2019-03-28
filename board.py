import numpy as np


class Board:
    """
    nxn grid board
    """
    character_map = {}

    def __init__(self, board_size=9):
        self._board_size = board_size
        self._board = np.zeros((board_size, board_size, 2))

    def legal_moves(self):
        legal_moves = [[i, j] for i in range(self._board_size) for j in range(self._board_size) if
                       ((self._board[i][j][0] == 0) and (self._board[i][j][1] == 0))]
        return legal_moves

    def update_board(self, move, player_id):
        if move in self.legal_moves():
            self._board[move[0]][move[1]][player_id] = 1
        else:
            raise ValueError(f"move:{move} is illegal! Please enter moves in the format [i, j]")

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
    print(board)


if __name__ == '__main__':
    main()
