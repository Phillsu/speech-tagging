# 儲存預測詞性
from build_dictionary import index_to_tag
from spilt_data_set import X_test
from tf_data_pipe_line import test_generator
from compile_rnn import model
import numpy as np

testing_preds = list()
# 儲存真實詞性tag
testing_true = list()

# 這裡使用兩個迴圈執行預測
# 第一個迴圈預測整個句子
for test in test_generator:
    words, tags = test
    testing_pred = model.predict(words)
    testing_pred_index = np.argmax(testing_pred, axis=-1)

    # 第二個迴圈將預測值以及真實標籤儲存起來
    for i in range(len(tags)):
        testing_preds.append([p for p in testing_pred_index[i] if p != 0])
        testing_true.append([p for p in tags[i].numpy() if p != 0])

# 印出第十筆來測試
print_index = 10
word = X_test[print_index]
pred = testing_preds[print_index]
true = testing_true[print_index]

pred_tag = [index_to_tag[t] for t in pred]
true_tag = [index_to_tag[t] for t in true]

print('Input words: \n', word)
print('Prediction: \n', pred_tag)
print('True: \n', true_tag)
