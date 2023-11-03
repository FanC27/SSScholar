import numpy as np
import file_handling as fh
import run_scholar as rs
from scipy.sparse import csr_matrix
import json

input_file = 'data/NYC/original_train/processed'
train_prefix = 'train'
test_prefix = 'test'
train_X, vocab, row_selector, train_ids = rs.load_word_counts(input_file, train_prefix)
test_X, _, row_selector, train_ids = rs.load_word_counts(input_file, test_prefix, vocab) # 得到[数据长度，词表长度], 词表token, 过滤器, 数据id
test_jsonlist = fh.read_jsonlist('data/NYC/original_train/NYC_test.jsonlist')
print(type(test_jsonlist))
pos_dict = {}
neg_dict = {}
new_list = []

def create_dict(input_file):
    for i, word_fre in enumerate(input_file):
        neg_count = 0
    # 得到每一行数据词的索引(list)
        if 100 <= i < 200:
            continue
        nonzero_row_indices = word_fre.nonzero()[1]
        row_ele = nonzero_row_indices.tolist()
        if i % 100 == 0:
            print("已经读取到第%d行数据" % i)
        # 遍历除了自己的每一行数据，找出正例
        for pos_i, pos_word in enumerate(input_file):
            if pos_i == i:
                continue
            pos_row_indics = pos_word.nonzero()[1]
            pos_row_ele = pos_row_indics.tolist()
            # 将正例元素的行写入字典中
            intersection = set(row_ele) & set(pos_row_ele)
            if len(pos_row_ele) - 1 <= len(row_ele) <= len(pos_row_ele) + 1:
                if len(pos_row_ele) - 1 <= len(intersection):
                    if i not in pos_dict:
                        pos_dict[i] = []
                    pos_dict[i].append(pos_i)
                if len(intersection) == 0:
                    neg_count = neg_count + 1
                    if neg_count == 50:
                        break
                    if i not in neg_dict:
                        neg_dict[i] = []
                    neg_dict[i].append(pos_i)
        if i not in pos_dict:
            pos_dict[i] = ['none']
        if i not in neg_dict:
            neg_dict[i] = ['none']

def write_query(input_file, dict):
    count = 0
    for key, value in dict.items():
        if len(value) == 1:
            count = count + 1
            if count <= 100:
                print(f"第{count}个:{key}的唯一标签是{value}")
                new_list.append(input_file[int(key)])
                new_list.append(input_file[int(value[0])])
    fh.write_jsonlist(new_list, 'new_test.jsonlist')

create_dict(train_X)

with open('/data/home/fcf/firstwork/at2vec/myScholar/data/NYC/original_train/pos_train_dict.json', 'w') as f:
    json.dump(pos_dict, f, indent=3)

with open('/data/home/fcf/firstwork/at2vec/myScholar/data/NYC/original_train/neg_train_dict.json', 'w') as f:
    json.dump(neg_dict, f, indent=3)