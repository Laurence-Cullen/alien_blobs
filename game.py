import contextlib

from agents import RandomPlayer
from board import Board


class Game:
    def __init__(self, board, player_one, player_two, game_length=20):
        self.board = board
        self.game_length = game_length
        self.player_one = player_one
        self.player_one.set_id(0)
        self.player_two = player_two
        self.player_two.set_id(1)
        self.game_length = game_length

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
            move = self.player_one.next_move(self.board)
            self.board.update_board(move, self.player_one.player_id)

            move = self.player_two.next_move(self.board)
            self.board.update_board(move, self.player_two.player_id)

            game_turn += 1


def main():
    board = Board()
    player_one = RandomPlayer(name='one')
    player_two = RandomPlayer(name='two')
    game = Game(board, player_one, player_two, game_length=40)
    print(game.board)
    game.play_game()
    print(game.board)
    print(game.player_one_score, game.player_two_score)


if __name__ == '__main__':
    main()
