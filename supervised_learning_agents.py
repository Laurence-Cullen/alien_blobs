from agents import Player
import pickle

# load the model from disk
forest_regressor = pickle.load(open('random_forest_model.sav', 'rb'))

class RandomForestPlayer(Player):
    def __init__(self, name, forest_regressor, player_id=None):
        super().__init__(name, player_id)
        self.forest_regressor = forest_regressor

    def board_value_assessment(self):



    def next_move(self, board):
        pass