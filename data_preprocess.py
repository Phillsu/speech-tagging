from install_nltk import tagged_sentences

words = []
tags = []

for sentence in tagged_sentences:
    X_sentence = []
    Y_sentence = []
    for entity in sentence:
        X_sentence.append(entity[0])
        Y_sentence.append(entity[1])

    words.append(X_sentence)
    tags.append(Y_sentence)

#print('sample words: ', words[0])
#print('sample tags: ', tags[0])