import tensorflow as tf

from constants import USE_LSTM


class RNN:
    def __init__(self):
        pass

    def add_recurrent_layers(self):
        raise NotImplemented

    def add_training_objectives(self):
        raise NotImplemented

    def create_model(self):
        self.add_recurrent_layers()
        self.add_training_objectives()