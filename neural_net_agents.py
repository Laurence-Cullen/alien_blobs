import pathlib
from copy import deepcopy

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import load_model

from agents import Player


class NeuralNetPlayer(Player):
    def __init__(self, name, board_size, model_file=None, trainable=True, player_id=None):
        super().__init__(name, player_id)

        self._trainable = trainable
        self._board_size = board_size

        # if model file is provided load it,
        # else use class method to set up architecture
        if model_file:
            self._model = load_model(model_file)
        else:
            self._model = self.initialise_model(board_size)

    @staticmethod
    def initialise_model(board_size):
        kernel_size = (3, 3)
        dropout = 0.7

        # build model topology
        model = Sequential()
        model.add(
            Conv2D(
                64,
                kernel_size=kernel_size,
                activation='relu',
                input_shape=(board_size, board_size, 2),
                data_format="channels_last",
                padding='same'
            )
        )

        model.add(BatchNormalization())
        model.add(Conv2D(30, kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(15, kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(300, activation='relu'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(5, activation='relu'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='tanh'))

        return model

    @staticmethod
    def load_from_files(file_paths, data, targets, new_target, row):

        for file_path in file_paths:
            board_array = np.load(file_path)

            # load in data from all 4 possible rotations of board array
            for k in range(4):
                data[row][:][:][:] = np.rot90(board_array, k=k)
                targets[row] = new_target
                row += 1

        return data, targets, row

    def load_data(self, data_dir):
        data_dir_path = pathlib.Path(data_dir)

        draw_board_files = list(data_dir_path.glob('-1/*.npy'))
        player_one_wins_board_files = list(data_dir_path.glob('-1/*.npy'))
        player_two_wins_board_files = list(data_dir_path.glob('-1/*.npy'))

        boards = len(draw_board_files) + len(player_one_wins_board_files) + len(player_two_wins_board_files)

        row = 0

        data = np.zeros((boards * 4, self._board_size, self._board_size, 2), dtype=np.int8)
        targets = np.zeros((boards * 4, 2), dtype=np.float16)

        # data, targets, row = self.load_from_files(
        #     file_paths=draw_board_files,
        #     data=data,
        #     targets=targets,
        #     new_target=[0.5, 0.5],
        #     row=row
        # )
        #
        # data, targets, row = self.load_from_files(
        #     file_paths=player_one_wins_board_files,
        #     data=data,
        #     targets=targets,
        #     new_target=[1, 0],
        #     row=row
        # )
        #
        # data, targets, row = self.load_from_files(
        #     file_paths=player_two_wins_board_files,
        #     data=data,
        #     targets=targets,
        #     new_target=[0, 1],
        #     row=row
        # )

        for draw_board_file in draw_board_files:
            board_array = np.load(draw_board_file)

            data[row][:][:][:] = board_array
            targets[row] = [0.5, 0.5]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=1)
            targets[row] = [0.5, 0.5]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=2)
            targets[row] = [0.5, 0.5]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=3)
            targets[row] = [0.5, 0.5]
            row += 1

        for player_one_wins_board_file in player_one_wins_board_files:
            board_array = np.load(player_one_wins_board_file)

            data[row][:][:][:] = board_array
            targets[row] = [1, 0]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=1)
            targets[row] = [1, 0]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=2)
            targets[row] = [1, 0]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=3)
            targets[row] = [1, 0]
            row += 1

        for player_two_wins_board_file in player_two_wins_board_files:
            board_array = np.load(player_two_wins_board_file)

            data[row][:][:][:] = board_array
            targets[row] = [1, 0]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=1)
            targets[row] = [1, 0]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=2)
            targets[row] = [1, 0]
            row += 1

            data[row][:][:][:] = np.rot90(board_array, k=3)
            targets[row] = [1, 0]
            row += 1

        return data, targets

    def train(self, data_dir):
        data, targets = self.load_data(data_dir)

        print(data, targets)
        epochs = 20
        batch_size = 64

        optimizer = keras.optimizers.Adam(lr=0.00001)

        self._model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=optimizer,
        )

        filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # train model
        self._model.fit(
            x=data,
            y=targets,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.2,
            callbacks=callbacks_list
        )

    def board_value_assessment(self, board):
        future_board_values = []
        legal_moves = board.legal_moves(board.board, board.board_size)
        for move in legal_moves:
            temp_board = deepcopy(board)
            temp_board.update_board(move, self.player_id)
            future_board_values.append(
                [move, self._model.predict(temp_board.board.reshape((1, board.board_size, board.board_size, 2)))]
            )

        return future_board_values

    def next_move(self, board):
        future_board_values = self.board_value_assessment(board)
        if self.player_id == 0:
            return max(future_board_values, key=lambda x: x[1][0][0])[0]

        if self.player_id == 1:
            return max(future_board_values, key=lambda x: x[1][0][1])[0]


def main():
    nn_player = NeuralNetPlayer(name='NN player', board_size=9)
    nn_player.train('data')


if __name__ == '__main__':
    main()
