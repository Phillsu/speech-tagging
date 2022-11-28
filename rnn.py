import tensorflow as tf
from build_dictionary import word_vocab_size, tag_vocab_size


class postag_rnn(tf.keras.Model):
    # model subcalssing要使用繼承，主要是要繼承tf.keras.Model這樣我們才能夠在後面使用model.ft, model.predict等函數
    def __init__(self, embedding_size, rnn_units):
        super().__init__()
        # 建立word embedding
        self.embedding = tf.keras.layers.Embedding(input_dim=word_vocab_size, output_dim=embedding_size)
        # 建立rnn模型
        self.rnn = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True)

        # 建立輸出層
        output_layer = tf.keras.layers.Dense(units=tag_vocab_size, activation='softmax')
        # 因為這是一個many to many的預測，也就是每個位置都要預測，所以要使用timedistributed，重複利用output_layer
        self.timedistributed = tf.keras.layers.TimeDistributed(output_layer)


    def call(self, x):
        embedded = self.embedding(x)
        hidden_states = self.rnn(embedded)
        outputs = self.timedistributed(hidden_states)

        return outputs
