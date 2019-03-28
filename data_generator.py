from league import League
import agents
import pathlib
import numpy as np

directory_names = ['-1', '0', '1']


class DataGenerator:
    def __init__(self, games):
        self._games = games

    def generate(self, save_folder, earliest_move=1, latest_move=None):
        save_directory = pathlib.Path(save_folder)

        for directory_name in directory_names:
            directory_path = save_directory / directory_name

            if not directory_path.exists():
                directory_path.mkdir(parents=True)

        for game_number, game in self._games.items():
            print(game.total_moves)

            if earliest_move >= game.total_moves:
                print('skipping game, not enough moves')
                continue

            if latest_move:
                last_move = min(latest_move, game.total_moves)
            else:
                last_move = game.total_moves

            for move_number in range(earliest_move, last_move):
                board = game.board_at_move(move_number).board

                np.save(save_directory / str(game.winner) / f'{game_number}_{move_number}.npy', board)


def main():
    league = League(players=[
        agents.ProximityRandomPlayer(name='proximity'),
        agents.RandomPlayer(name='random'),
    ])
    league.play_games(number_of_games=10, turns_per_game=10)

    print(league)
    print(league.games)

    datagen = DataGenerator(league.games)
    datagen.generate(save_folder='data')


if __name__ == '__main__':
    main()
