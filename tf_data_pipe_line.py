from spilt_data_set import X_test, X_train, y_train, y_test
from build_dictionary import word_to_index, tag_to_index
import tensorflow as tf

train_tfdata = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_tfdata = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 將word和tag使用' '來分開
def encode(word, tag):
    word = [word_to_index[t] for t in word.numpy().decode().split(' ')]
    tag = [tag_to_index[t] for t in tag.numpy().decode().split(' ')]
    return word, tag


# 使用tf.py_function將encode轉換為tf.data
def tf_encode(word, tag):
    return tf.py_function(encode, [word, tag], [tf.int32, tf.int32])


buffer_size =320
batch_size = 32
padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
train_generator = train_tfdata.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(buffer_size).padded_batch(batch_size,padded_shapes=padded_shapes).repeat()
test_generator = test_tfdata.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE).padded_batch(batch_size,padded_shapes=padded_shapes)

x = iter(train_generator)
tmp_inp =next(x)
print(tmp_inp)
