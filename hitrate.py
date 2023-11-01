import numpy as np
import file_handling as fh
import argparse

# 假设您已经有query向量列表，label向量列表和数据库向量列表
# query_vectors是包含10个query向量的数组，形状为[10, 向量维数]
# label_vectors是包含10个label向量的数组，形状为[10, 向量维数]
# database_vectors是包含100个数据库向量的数组，形状为[100, 向量维数]

# 假设相似度度量是余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

parser = argparse.ArgumentParser(description='get the hitrate score')
parser.add_argument('--struct-model', type=str, default='t2vec',
                  help='find the struct model name')
parser.add_argument('--dataset', type=str, default='NYC',
                  help='get the dataset(NYC or TKY)')
parser.add_argument('--mode', type=str, default='test',
                  help='get the mode(train or test)')
parser.add_argument('--contrastive', action='store_true', dest='contrastive', default=True,
                  help='get the contrastive result')
parser.add_argument('--no-contrastive', action='store_false', dest='contrastive', default=False)
parser.add_argument('--merge', action='store_true', dest='merge', default=True,
                  help='get the merge result')
parser.add_argument('--no-merge', action='store_false', dest='merge', default=False,
                  help='get the merge result')
parser.add_argument('--k', type=int, default=5,
                  help='get the topic number')

args = parser.parse_args()

test_data = fh.get_struct_vec(args.mode, args.struct_model, args.dataset, None, args.contrastive, args.merge, args.k)

query_vectors = test_data[:100]
label_vectors = test_data[100:200]
database_vectors = test_data[200:2200]
count = 0
for i in range(5):
    print(query_vectors[i][:3])
    print(label_vectors[i][:3])
    print(database_vectors[i][:3])
    q_l_similarity = cosine_similarity(query_vectors[i], label_vectors[i])
    print(q_l_similarity)
    q_d_similarity = cosine_similarity(query_vectors[i], database_vectors[i])
    print(q_d_similarity)
    if q_l_similarity > q_d_similarity:
        print('命中！！！！')
        count = count + 1
    print('*' * 20)

# 计算query和label的相似度 每个query和label的相似度
similarity_scores_query_label = [cosine_similarity(query, label) for query, label in zip(query_vectors, label_vectors)]


# # 计算query和数据库的相似度，并对相似度排序
# similarity_scores_query_database = []
# for query in query_vectors:
#     similarities = [cosine_similarity(query, database) for database in database_vectors]
#     similarities = sorted(similarities, reverse=True)  # 对相似度进行逆行排序
#     similarity_scores_query_database.append(similarities)

# # 计算命中率@10
# hit_rate_10 = 0
# for i in range(len(query_vectors)):
#     label_similarity = similarity_scores_query_label[i]
#     database_similarity = similarity_scores_query_database[i][:100]  # 只取前10个
#     for sim in database_similarity:
#         if label_similarity == sim:  # 检查相似度是否相等
#             hit_rate_10 += 1
#             break  # 找到匹配项后跳出循环

# hit_rate_10 /= len(query_vectors)
# print(f"Hit rate@10: {hit_rate_10}")