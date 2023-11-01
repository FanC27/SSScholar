import numpy as np
import torch
from scipy.stats import cosine
import file_handling as fh
import argparse
import pathlib

device_1 = torch.device("cuda:1")

def main():
    parser = argparse.ArgumentParser(description='get the meanrank score')
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
    
    # 融合后的meanrank
    # test_data = fh.get_struct_vec(args.mode, args.struct_model, args.dataset, None, args.contrastive, args.merge, args.k)
   
    # file_path = fh.maketxts(args.struct_model, args.dataset, args.contrastive, args.merge, args.k)

    # meanrank(test_data, file_path)

    # 直接拼接的meanrank
    # activate_test_data = fh.get_struct_vec(args.mode, args.struct_model, args.dataset, None, args.contrastive, args.merge, args.k)
    # struct_test_data = fh.get_struct_vec('train', args.struct_model, args.dataset, 'test', args.contrastive, args.merge, args.k)
    activate_test_data = fh.read_npy('theta.test_1.npz')
    query = activate_test_data[:100]
    label = activate_test_data[100:200]
    database = activate_test_data[200:]

    # struct_query = struct_test_data[:100]
    # struct_label = struct_test_data[100:200]
    # struct_database = struct_test_data[200:]

    # 将两个数组水平拼接
    # query = np.hstack((activate_query, struct_query))
    # label = np.hstack((activate_label, struct_label))
    # database = np.hstack((activate_database, struct_database))
    
    subset_sizes = [400, 800, 1200, 1600, 2000]  #
    all_ranks = []
    for size in subset_sizes:
        database_vectors_subset = database[:size]  # 取数据库的前size个向量作为子集
        ranks = calculate_rank(query, label, database_vectors_subset)
        
        count = sum(ranks)
        length = len(ranks)
        mean_rank = count / length
        all_ranks.append(mean_rank)

    print("不同大小比例的数据库中查询向量对应的真实匹配向量的排名:")
    for i, size in enumerate(subset_sizes):
        print(f"数据库前{size}个向量的排名: {all_ranks[i]}")




def calculate_similarity(query_vector, database_vectors):
    """
    计算查询向量和数据库向量之间的相似度（余弦相似度）

    参数：
    query_vector: numpy数组，查询向量
    database_vectors: numpy数组，包含数据库中的所有向量

    返回：
    similarities: numpy数组，包含查询向量与每个数据库向量之间的相似度值
    """
    normalized_query = query_vector / np.linalg.norm(query_vector)
    # 将数据库向量标准化，即将所有向量转换为单位向量
    if database_vectors.ndim == 1:
        normalized_database = database_vectors / np.linalg.norm(database_vectors)
    else:
        normalized_database = database_vectors / np.linalg.norm(database_vectors, axis=1, keepdims=True)
    # 计算查询向量与数据库中所有向量的点积，得到相似度值
    similarities = np.dot(normalized_database, normalized_query)
    return similarities

def calculate_rank(query_vectors, label_vectors, database_vectors_subset):
    """
    计算查询向量对应的真实匹配向量在数据库子集中的排名

    参数：
    query_vectors: numpy数组，包含所有查询向量
    label_vectors: numpy数组，包含所有真实匹配向量（groundtruth）
    database_vectors_subset: numpy数组，包含数据库子集中的所有向量

    返回：
    ranks: list，包含每个查询向量对应的排名
    """
    num_queries = query_vectors.shape[0]
    ranks = []

    for i in range(num_queries):
        query_vector = query_vectors[i]
        label_vector = label_vectors[i]  # 获取查询向量对应的真实匹配向量
        label_similarity = calculate_similarity(query_vector, label_vector)
        database_similarities = calculate_similarity(query_vector, database_vectors_subset)
        rank = np.sum(database_similarities >= label_similarity) + 1  # 计算label在database中的排名
        ranks.append(rank)

    return ranks

def meanrank(test_data, output_file):
    query = test_data[:100]
    label = test_data[100:200]
    database = test_data[200:2200]

    output_file = open(output_file, 'a')
    

    subset_sizes = [400, 800, 1200, 1600, 2000]  #
    all_ranks = []
    for size in subset_sizes:
        database_vectors_subset = database[:size]  # 取数据库的前size个向量作为子集
        ranks = calculate_rank(query, label, database_vectors_subset)
        output_file.write(str(ranks))
        count = sum(ranks)
        length = len(ranks)
        mean_rank = count / length
        all_ranks.append(mean_rank)

    print("不同大小比例的数据库中查询向量对应的真实匹配向量的排名:")
    for i, size in enumerate(subset_sizes):
        print(f"数据库前{size}个向量的排名: {all_ranks[i]}")

    
    output_file.write("不同大小比例的数据库中查询向量对应的真实匹配向量的排名:\n")
    for i, size in enumerate(subset_sizes):
        output_file.write(f"数据库前{size}个向量的排名: {all_ranks[i]}\n")

    print(f"结果已保存至 {output_file} 文件中。")

if __name__ == '__main__':
    main()