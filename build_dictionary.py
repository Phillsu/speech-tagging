from data_preprocess import words, tags
from keras.preprocessing.text import Tokenizer
import numpy as np


def set_vocab(words):
    set_words = list(set([t1 for t2 in words for t1 in t2]))
    words_index = np.array(range(len(set_words))) + 1
    # 這裡+1表示index從1開始，因為後續需要將長度不足的句子進行補齊(padding)
    word_to_index = dict(zip(set_words, words_index))
    index_to_word = dict(zip(words_index, set_words))
    return word_to_index, index_to_word


word_to_index, index_to_word = set_vocab(words)
tag_to_index, index_to_tag = set_vocab(tags)

#print('---------------------------------')
#print('tag to index:')
#print(tag_to_index)
#print('---------------------------------')
#print('word to index:')
#print(word_to_index)

word_vocab_size = list(index_to_word)[-1] +1
tag_vocab_size = list(index_to_tag)[-1] +1
#print('word vocabulary size:', word_vocab_size)
#print('tag vocabulary size:', tag_vocab_size)
