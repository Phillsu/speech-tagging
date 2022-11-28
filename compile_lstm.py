import tensorflow as tf
from lstm import postag_lstm
from spilt_data_set import X_train
from tf_data_pipe_line import train_generator, test_generator, batch_size

embedding_size = 256
rnn_units = 512

model = postag_lstm(embedding_size, rnn_units)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_generator, epochs=2, validation_data=test_generator, steps_per_epoch=len(X_train) // batch_size + 1)

loss, accuracy = model.evaluate(test_generator)
print("test dataset's accuracy: {:.2f}".format(accuracy))
