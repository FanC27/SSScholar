import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 读取GloVe预训练向量文件
glove_vectors = {}
with open('/data/home/fcf/firstwork/ltc/scholar/glove.6B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_vectors[word] = vector

avgcos = 0

# 读取包含单词的txt文件并计算 avgcos(k)
with open('/data/home/fcf/firstwork/ltc/scholar/output/topics.txt', 'r') as f:
    for line in f:
        words = line.split()
        vectors = [glove_vectors[word] for word in words if word in glove_vectors]

        sum_cosine_sim = 0
        count = 0

        for i in range(len(vectors) - 1):
            for j in range(i + 1, len(vectors)):
                v1 = vectors[i]
                v2 = vectors[j]
                cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sum_cosine_sim += cosine_sim
                count += 1
        avg_cosine_sim = sum_cosine_sim / count
        avgcos = avgcos + avg_cosine_sim
        print(f"Avgcos(k) for this line: {avg_cosine_sim}")
print(avgcos / 5)