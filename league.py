import cProfile
import collections
import random
from itertools import combinations

import elo
from agents import RandomPlayer
from board import Board
from game import Game


class LeagueMember:
    def __init__(self, player, initial_elo=1500):
        self._player = player
        self.elo = initial_elo

    @property
    def player(self):
        return self._player


class League:
    def __init__(self, players, board_size=9):
        self._players = players
        self._members = {}
        self._game_history = []
        self._board_size = board_size

        for player in self._players:
            self._members[player.name] = LeagueMember(player)

        self._match_ups = list(combinations(list(self._members.keys()), 2))

    def add_player(self, player):
        self._players.append(player)

    def play_games(self, games=10):
        """
        Pick random members of the league and get them to play each other,
        updating their ELO after each game is completed.
        """

        total_games = games

        while games > 0:
            if games % (int(total_games / 10) + 1) == 0:
                print(f'{((total_games - games) / total_games) * 100:.1f} percent complete')

            member_one, member_two = self.get_random_members()

            game = Game(
                player_one=member_one.player,
                player_two=member_two.player,
                board=Board(board_size=self._board_size)
            )

            game.play_game()

            score_one = game.player_one_score
            score_two = game.player_two_score

            print(score_one, score_two)

            try:
                normalised_score_one = score_one / (score_one + score_two)
                normalised_score_two = score_two / (score_one + score_two)
            except ZeroDivisionError:
                normalised_score_one = 0.5
                normalised_score_two = 0.5

            elo.update_elo(
                member_one=member_one,
                member_two=member_two,
                score_one=normalised_score_one,
                score_two=normalised_score_two
            )

            self._game_history.append(game)

            games -= 1

    def get_random_members(self):
        """
        Pick two different members of the league.
        """

        return random.sample(list(self._members.values()), 2)

    def __str__(self):
        lines = []
        sorted_league = sorted(self._members.items(), key=lambda items: items[1].elo, reverse=True)
        sorted_league = collections.OrderedDict(sorted_league)

        for name, member in sorted_league.items():
            lines.append(f'{name}: {member.elo:.2f}')

        return '\n'.join(lines)


def main():
    league = League(players=[
        RandomPlayer(name='rand1'),
        RandomPlayer(name='rand2'),
    ])
    league.play_games(games=10000)

    print(league)


if __name__ == '__main__':
    # 32.7 seconds current run time
    cProfile.run('main()')

    # main()
