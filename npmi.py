import math
from collections import Counter

# 计算NPMI值
def compute_npmi(tokens):
    total_tokens = sum(tokens.values())
    ppmi = {}
    for token, count in tokens.items():
        ppmi[token] = count / total_tokens

    sum_npmi = 0
    num_pairs = 0
    for token1, count1 in tokens.items():
        for token2, count2 in tokens.items():
            if token1 != token2:
                pmi = math.log2((count1 * total_tokens) / (tokens[token1] * tokens[token2]))
                npmi = pmi / -math.log2(ppmi[token1] * ppmi[token2])
                sum_npmi += npmi
                num_pairs += 1
    if num_pairs > 0:
        return sum_npmi / num_pairs
    else:
        return 0

# 读取包含文本文件路径的txt文件，并将每一行进行分词
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 逐行读取文本文件内容
    return [line.strip().split() for line in lines]  # 使用空格分割每一行的token

if __name__ == '__main__':
    file_path = '/data/home/fcf/firstwork/ltc/scholar/output/topics.txt'  # 修改为您的文件路径
    token_lists = read_txt_file(file_path)  # 读取分词后的每一行内容

    for idx, tokens in enumerate(token_lists, start=1):  # 遍历每一行的分词内容
        token_count = Counter(tokens)  # 计算每个分词的出现次数
        npmi_value = compute_npmi(token_count)  # 计算NPMI值
        print(f"NPMI value for document {idx}: {npmi_value}")  # 输出每一行的NPMI值列表