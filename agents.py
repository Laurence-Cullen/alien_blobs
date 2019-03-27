import random
import keras
from keras.models import model_from_yaml, Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense


class Player:
    def __init__(self, name):
        self._name = name

    def play_move(self, board, player_id):
        pass

    def next_move(self, board, player_id):
        pass

    @property
    def name(self):
        return self._name


class RandomPlayer(Player):
    """
    Plays on a random clear part of the board.
    """

    def next_move(self, board, player_id):
        return random.choice(board.legal_moves())


class ProximityRandomPlayer(Player):
    """
    Plays on a random part of the board that is adjacent to opponent pieces if possible.
    """

    def next_move(self, board, player_id):
        proximity_moves = board.legal_moves_adjacent_to_player(1 - player_id)

        if len(proximity_moves) > 0:
            return random.choice(proximity_moves)
        else:
            return random.choice(board.legal_moves())


class NeuralNetPlayer(Player):
    def __init__(self, name, board, architecture_file=None, weights_file=None, trainable=True):
        super().__init__(name)

        self._trainable = trainable

        # if architecture file is provided load it,
        # else use class method to set up architecture
        if architecture_file:
            self._model = model_from_yaml(architecture_file)
        else:
            self._model = self.initialise_model(board)

        # if weights file provided load it, otherwise weights will be randomised
        if weights_file:
            self._model.load_weights(weights_file)

    @staticmethod
    def initialise_model(board):
        kernel_size = (3, 3)
        dropout = 0.2

        # build model topology
        model = Sequential()
        model.add(
            Conv2D(
                64,
                kernel_size=kernel_size,
                activation='relu',
                input_shape=board.board.shape,
                data_format="channels_last"
            )
        )
        model.add(BatchNormalization())
        model.add(Conv2D(30, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='sigmoid'))

        return model