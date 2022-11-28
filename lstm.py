import tensorflow as tf
from build_dictionary import word_vocab_size, tag_vocab_size


class postag_lstm(tf.keras.Model):
    def __init__(self, embedding_size, lstm_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=word_vocab_size, output_dim=embedding_size)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        output_layer = tf.keras.layers.Dense(units=tag_vocab_size, activation='softmax')
        self.timedistributed = tf.keras.layers.TimeDistributed(output_layer)

    def call(self, x):
        embedded = self.embedding(x)
        hidden_states = self.lstm(embedded)
        outputs = self.timedistributed(hidden_states)

        return outputs
