import os
import json
import codecs
import pickle
import numpy as np
from scipy import sparse
import torch
import pathlib

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file, encoding='utf-8')
    return data


def read_jsonlist(input_filename):
    data = []
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line, encoding='utf-8'))
    return data


def write_jsonlist(list_of_json_objects, output_filename, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines


def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    if add_newlines:
        lines = '\n'.join(lines)
        if add_final_newline:
            lines += '\n'
    else:
        lines = ''.join(lines)
        if add_final_newline:
            lines[-1] += '\n'

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)


def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


def load_sparse(input_filename):
    npy = np.load(input_filename, allow_pickle=True)  # 读取npy文件
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()

def read_npy(input_filename):
    data = np.load(input_filename)
    title_name = data.files
    return data[title_name[0]]

def get_struct_vec(mode, model_name, dataset, data_type, contrastive, merge, topic_nums):
    if mode == 'train':
        input_filename = f'structure_data/{model_name}/{dataset}/{dataset}_{data_type}.npz'
    elif mode == 'test':
        if contrastive and merge:
            input_filename = f'output/{model_name}/{dataset}/test/theta_contrastive_merge_{topic_nums}.npz'
        elif contrastive:
            input_filename = f'output/{model_name}/{dataset}/test/theta_contrastive_{topic_nums}.npz'
        elif merge:
            input_filename = f'output/{model_name}/{dataset}/test/theta_merge_{topic_nums}.npz'
        else:
            input_filename = f'output/{model_name}/{dataset}/test/theta_{topic_nums}.npz'
        
    data = read_npy(input_filename)
    n_items, emb_size = data.shape
    # vector = torch.tensor(data)
    print("Loaded %d struct_trajectory with %d embedding_size" % (n_items, emb_size))
    print(f"读取文件:{input_filename}")
    return data

def maketxts(model_name, dataset, contrastive, merge, topic_nums):
    if contrastive and merge:
        output_file = f'output/{model_name}/{dataset}/test/result_contrastive_merge_{topic_nums}.txt'
    elif contrastive:
        output_file = f'output/{model_name}/{dataset}/test/result_contrastive_{topic_nums}.txt'
    elif merge:
        output_file = f'output/{model_name}/{dataset}/test/result_merge_{topic_nums}.txt'
    else:
        output_file = f'output/{model_name}/{dataset}/test/result_{topic_nums}.txt'

    pathlib.Path(output_file).touch()
    print(f"保存当前结果的txt文件已成功创建，路径为: {output_file}")

    return output_file