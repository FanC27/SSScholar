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
avg = 0
# 读取包含单词的txt文件并计算 stdcos(k)
with open('/data/home/fcf/firstwork/ltc/scholar/output/topics.txt', 'r') as f:
    for line in f:
        words = line.split()
        vectors = [glove_vectors[word] for word in words if word in glove_vectors]

        avg_cosine_sim = sum(
            np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            for i in range(len(vectors) - 1)
            for j in range(i + 1, len(vectors))
        ) / len(vectors)

        sum_squared_diff = sum(
            (np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])) - avg_cosine_sim) ** 2
            for i in range(len(vectors) - 1)
            for j in range(i + 1, len(vectors))
        )
        count = len(vectors) * (len(vectors) - 1) / 2

        std_cosine_sim = np.sqrt(sum_squared_diff / count)
        avg = std_cosine_sim + avg
        print(f"Stdcos(k) for this line: {std_cosine_sim}")

print(avg / 5)