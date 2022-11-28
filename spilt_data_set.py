from data_preprocess import words,tags
from sklearn.model_selection import train_test_split
#在切割之前用' '將詞跟詞合併，詞性跟詞性合併
X_train, X_test, y_train, y_test = train_test_split([' '.join(w) for w in words], [' '.join(w) for w in tags], test_size=0.2)

print('Training data size: %d' %len(X_train))
print('Testing data size: %d' %len(X_test))
