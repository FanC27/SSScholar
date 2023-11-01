import numpy as np
import json
import file_handling as fh
import os
# data = np.load('/data/home/fcf/firstwork/ltc/scholar/data/poi_review.pkl', allow_pickle=True)
# print(type(data['review']))
# reviews = data['review']
# reviews_list = reviews.values.tolist()
# print(len(reviews_list))
# # 打印每行评论
# # for i, review in enumerate(reviews_list[15000:15002]):
# #     print(f"Review {i+1}: {review}")

# file_name = "documents.txt"
# with open(file_name, 'w', encoding='utf-8') as file:
#     for document in reviews_list[15000:17000]:
#         file.write(document.replace('\n', '') + "\n")

# # 读取 JSON Lines 文件
# data = []
# with open('/data/home/fcf/firstwork/ltc/scholar/data/imdb/train.jsonlist', 'r', encoding='utf-8') as file:
#     for line in file:
#         data.append(json.loads(line))

# print(len(data))

# # 处理数据
# # for item in data[:5]:
# #     # 在这里进行您想要的操作
# #     print(item['id'])

# for i in range(15000):
#     data[i]['text'] = reviews_list[i].replace('\n', '')

# with open('/data/home/fcf/firstwork/ltc/scholar/data/train_file.jsonlist', 'w', encoding='utf-8') as file:
#     for item in data[:15000]:
#         json.dump(item, file, ensure_ascii=False)
#         file.write('\n')

# data = np.load('output/beta.npz', allow_pickle=True)
# print(data['beta'].shape)
# count = 0
# for i in data['beta']:
#     print(sum(i))

# query = fh.get_struct_vec('t2vec', 'NYC', 'query')
# label = fh.get_struct_vec('t2vec', 'NYC', 'label')
# database = fh.get_struct_vec('t2vec', 'NYC', 'database')

# NYC_test = np.concatenate((query, label, database), axis=0)

# np.savez('NYC_test.npz', NYC_test = NYC_test)

fh.makedirs(os.path.join('output','train','sss.npz'))