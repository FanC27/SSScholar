import json
import file_handling as fh
import random
# merge_data
# query = fh.read_jsonlist('data/data_NYC/query/NYC_query.jsonlist')
# label = fh.read_jsonlist('data/data_NYC/label/NYC_label.jsonlist')
# database = fh.read_jsonlist('data/data_NYC/database/NYC_database.jsonlist')
# merge_list = query + label + database
# fh.write_jsonlist(merge_list, 'test.jsonlist')

train = fh.read_jsonlist('data/data_NYC/original_train/NYC_train.jsonlist')
with open('data/data_NYC/data_classification.json', 'r') as f:
    my_dict = json.load(f)
for m, lines in enumerate(train):
    my_list = lines['text'].split()
    print(my_list)
    # 创造训练样本正例
    # for i in range(len(my_list)):
    #     for key, value in my_dict.items():
    #         if my_list[i] in value:
    #             index = value.index(my_list[i])
    #             if index < len(value) - 1:
    #                 my_list[i] = value[index + 1]
    #             else:
    #                 my_list[i] = value[0]
    #             break

    # 创造训练样本负例
    for word in my_list:
        for key, value in my_dict.items():
            if word in value:
                new_key = random.choice(list(my_dict.keys()))
                while new_key == key:
                    new_key = random.choice(list(my_dict.keys()))
                my_list[my_list.index(word)] = random.choice(my_dict[new_key])
                break
    train[m]['text'] = " ".join(word for word in my_list)
fh.write_jsonlist(train, 'NYC_neg_train.jsonlist')
