import contextlib

from agents import RandomPlayer
from board import Board


class Game:
    def __init__(self, board, player_one, player_two, game_length=20):
        self.board = board

        # record of the sequential moves making up the game in form {move_num: [i, j, player_id]}
        self._moves = {}

        # tracks how many moves have been made
        self._move_counter = 0

        self.game_length = game_length
        self.player_one = player_one
        self.player_one.set_id(0)
        self.player_two = player_two
        self.player_two.set_id(1)
        self.game_length = game_length

        self.winner = None

    def player_score(self, player_id):
        player_score = 0
        for [i, j] in [[i, j] for i in range(self.board.board_size) for j in range(self.board.board_size) if
                       self.board.board[i][j][1 - player_id] == 1]:

            # look over all neighbouring squares around opponent piece,
            # if they are all 1 add one to players score
            with contextlib.suppress(IndexError):
                if self.board.board[i + 1][j][player_id] == 0:
                    continue

            with contextlib.suppress(IndexError):
                if self.board.board[i - 1][j][player_id] == 0:
                    continue

            with contextlib.suppress(IndexError):
                if self.board.board[i][j + 1][player_id] == 0:
                    continue

            with contextlib.suppress(IndexError):
                if self.board.board[i][j - 1][player_id] == 0:
                    continue

            player_score += 1

        return player_score

    @property
    def player_one_score(self):
        return self.player_score(self.player_one.player_id)

    @property
    def player_two_score(self):
        return self.player_score(self.player_two.player_id)

    def play_game(self):
        game_turn = 0

        while game_turn < self.game_length:
            self.play_turn(self.player_one)
            self.play_turn(self.player_two)

            game_turn += 1

    def play_turn(self, player):
        move = player.next_move(self.board)
        self.board.update_board(move, player.player_id)
        self._moves[self._move_counter] = [move[0], move[1], player.player_id]
        self._move_counter += 1

    def board_at_move(self, move_num):
        """
        Reconstruct the board state at a particular move, building the board from the history of
        moves up to this point.
        """

        board = Board(board_size=self.board.board_size)

        for i in range(move_num):
            board.update_board(move=[self._moves[i][0], self._moves[i][1]], player_id=self._moves[i][2])

        return board


def main():
    board = Board()
    player_one = RandomPlayer(name='one')
    player_two = RandomPlayer(name='two')
    game = Game(board, player_one, player_two, game_length=40)
    print(game.board)
    game.play_game()
    print(game.board)
    print(game.player_one_score, game.player_two_score)

    print(game.board_at_move(10))


if __name__ == '__main__':
    main()
