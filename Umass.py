import math

# 从txt文件中读取主题词分布数据
with open('/data/home/fcf/firstwork/ltc/scholar/output/topic2word_prob.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行数据并计算UMass分数
def compute_umass(lines, epsilon=0):
    umass_sum = 0.0
    for line in lines:
        word_dist = line.split()  # 拆分每行数据
        for i, p_wi in enumerate(word_dist, start=1):
            for j, p_wj in enumerate(word_dist):
                if i != j:
                    p_wi_wj = float(p_wi) * float(p_wj)
                    umass_sum += math.log((p_wi_wj + epsilon) / float(p_wj))
    umass = 2 / (len(lines) * (len(lines) - 1)) * umass_sum
    return umass

# 计算UMass分数
umass_score = compute_umass(lines)
print(f"UMass score: {umass_score}")