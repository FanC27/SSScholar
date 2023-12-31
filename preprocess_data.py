import os
import re
import sys
import string
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat

import file_handling as fh

"""
Convert a dataset into the required format (as well as formats required by other tools).
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document text.
Any other field can be used as a label (specified with the --label option).
If training and test data are to be processed separately, the same input directory should be used
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def main(args):
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default=None,
                      help='field(s) to use as label (comma-separated): default=%default')
    parser.add_option('--test', dest='test', default=None,
                      help='Test data (test.jsonlist): default=%default')
    parser.add_option('--train-prefix', dest='train_prefix', default='train',
                      help='Output prefix for training data: default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default='test',
                      help='Output prefix for test data: default=%default')
    parser.add_option('--stopwords', dest='stopwords', default='snowball',
                      help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
    parser.add_option('--min-doc-count', dest='min_doc_count', default=0,
                      help='Exclude words that occur in less than this number of documents')
    parser.add_option('--max-doc-freq', dest='max_doc_freq', default=1.0,
                      help='Exclude words that occur in more than this proportion of documents')
    parser.add_option('--keep-num', action="store_true", dest="keep_num", default=False,
                      help='Keep tokens made of only numbers: default=%default')
    parser.add_option('--keep-alphanum', action="store_true", dest="keep_alphanum", default=False,
                      help="Keep tokens made of a mixture of letters and numbers: default=%default")
    parser.add_option('--strip-html', action="store_true", dest="strip_html", default=False,
                      help='Strip HTML tags: default=%default')
    parser.add_option('--no-lower', action="store_true", dest="no_lower", default=False,
                      help='Do not lowercase text: default=%default')
    parser.add_option('--min-length', dest='min_length', default=3,
                      help='Minimum token length: default=%default')
    parser.add_option('--vocab-size', dest='vocab_size', default=None,
                      help='Size of the vocabulary (by most common, following above exclusions): default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random integer seed (only relevant for choosing test set): default=%default')

    (options, args) = parser.parse_args(args)

    train_infile = args[0]
    output_dir = args[1]

    test_infile = options.test
    train_prefix = options.train_prefix
    test_prefix = options.test_prefix
    label_fields = options.label
    min_doc_count = int(options.min_doc_count)
    max_doc_freq = float(options.max_doc_freq)
    vocab_size = options.vocab_size
    stopwords = options.stopwords
    if stopwords == 'None':
        stopwords = None
    keep_num = options.keep_num
    keep_alphanum = options.keep_alphanum
    strip_html = options.strip_html
    lower = not options.no_lower
    min_length = int(options.min_length)
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_data(train_infile, test_infile, output_dir, train_prefix, test_prefix, min_doc_count, max_doc_freq, vocab_size, stopwords, keep_num, keep_alphanum, strip_html, lower, min_length, label_fields=label_fields)


def preprocess_data(train_infile, test_infile, output_dir, train_prefix, test_prefix, min_doc_count=0, max_doc_freq=1.0, vocab_size=None, stopwords=None, keep_num=False, keep_alphanum=False, strip_html=False, lower=True, min_length=3, label_fields=None):

    if stopwords == 'mallet':
        print("Using Mallet stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        print("Using snowball stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = fh.read_text(os.path.join('stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}  # 停词列表

    print("Reading data files")
    train_items = fh.read_jsonlist(train_infile)  # 读取训练数据的每一行，存在列表中
    n_train = len(train_items)  # 获得训练集长度
    print("Found {:d} training documents".format(n_train))  # 找到n条训练集
    for line in train_items:
        # if line["text"].indexof("Caf\x1a\x1a") != -1:
        str_new = line["text"].replace("Caf\x1a\x1a", "Cafe")
        line["text"] = str_new
    # train_items[466153]["text"] = "Cafe Cafe Cafe Cafe Cafe Cafe Cafe Cafe"


    if test_infile is not None:  # 如果存在测试
        test_items = fh.read_jsonlist(test_infile)  # 读取测试数据的每一行，存在列表中
        n_test = len(test_items)  # 获得测试集长度
        for line in test_items:
            # if line["text"].indexof("Caf\x1a\x1a") != -1:
            str_new = line["text"].replace("Caf\x1a\x1a", "Cafe")
            line["text"] = str_new
        print("Found {:d} test documents".format(n_test))  # 找到n条测试集
    else:
        test_items = []  # 不存在测试集，就为空
        n_test = 0

    all_items = train_items + test_items
    n_items = n_train + n_test

    "没有情感标签，不构建"
    label_lists = {}
    if label_fields is not None:
        if ',' in label_fields:
            label_fields = label_fields.split(',')
        else:
            label_fields = [label_fields]
        for label_name in label_fields:
            label_set = set()
            for i, item in enumerate(all_items):
                if label_name is not None:
                    label_set.add(item[label_name])
            label_list = list(label_set)
            label_list.sort()
            n_labels = len(label_list)
            print("Found label %s with %d classes" % (label_name, n_labels))
            label_lists[label_name] = label_list
    else:
        label_fields = []

    # make vocabulary 构建词表（列表）
    train_parsed = []
    test_parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()  # 统计单词个数
    doc_counts = Counter()  # 统计文档个数
    count = 0

    vocab = None
    for i, item in enumerate(all_items):  # 得到全部数据的索引、数据
        if i % 1000 == 0 and count > 0:
            print(i)

        text = item['text']  # 得到每个文档的文本内容
        '获得当前文本的token列表'
        tokens, _ = tokenize(text, strip_html=strip_html, lower=lower, keep_numbers=keep_num, keep_alphanum=keep_alphanum, min_length=min_length, stopwords=stopword_set, vocab=vocab)

        # store the parsed documents
        if i < n_train:  # 如果是训练数据
            train_parsed.append(tokens)  # 把该文档的所有token加到列表中
        else:
            test_parsed.append(tokens)

        # keep track fo the number of documents with each word
        word_counts.update(tokens)  # 所有文档的每个token出现的频次 [a : 3, b : 4, c : 2, d : 6 ...]
        doc_counts.update(set(tokens))  # 每个单词总共出现在几篇文档中 [a : 1, b : 1, c : 1, d : 1 ...]

    print("Size of full vocabulary=%d" % len(word_counts))  # 输出：当前所有文档共含有多少不同的词，也就是词表（one-hot向量）

    print("Selecting the vocabulary")
    most_common = doc_counts.most_common()  # 对所有词频进行排序，由大到小 每个单词总共出现在几篇文档中
    words, doc_counts = zip(*most_common)  # 解压，words=单词， doc_counts=单词在不同文档出现的频次
    doc_freqs = np.array(doc_counts) / float(n_items)  # 该单词出现的文档数 / 所有文档数 = 该单词在一个文档中出现的频率
    vocab = [word for i, word in enumerate(words) if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq]  # 如果该单词数>0且词频<=1 就放入词表中
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]  # 频率>1的为0
    if max_doc_freq < 1.0:
        print("Excluding words with frequency > {:0.2f}:".format(max_doc_freq), most_common)

    print("Vocab size after filtering = %d" % len(vocab))  # 过滤后的词表长度
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):  # 如果所得词表长度>2000
            vocab = vocab[:int(vocab_size)]  # 只取前两千个词（最高频率的词看作重点）

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", ' '.join(vocab[:10]))  # 出现频率最高的词
    vocab.sort()  # 按字母进行排序

    fh.write_to_json(vocab, os.path.join(output_dir, train_prefix + '.vocab.json'))

    """
    train_X_sage = 所有文档的词频矩阵
    tr_widx = 词表序号
    vocab_for_sage = 词表内容
    """
    train_X_sage, tr_aspect, tr_no_aspect, tr_widx, vocab_for_sage = process_subset(train_items, train_parsed, label_fields, label_lists, vocab, output_dir, train_prefix)
    if n_test > 0:
        test_X_sage, te_aspect, te_no_aspect, _, _= process_subset(test_items, test_parsed, label_fields, label_lists, vocab, output_dir, test_prefix)

    train_sum = np.array(train_X_sage.sum(axis=0))  # [1 * 455] 词表中的词出现的所有次数
    print("%d words missing from training data" % np.sum(train_sum == 0))

    if n_test > 0:
        test_sum = np.array(test_X_sage.sum(axis=0))  # [1 * 455] 词表中的词出现的所有次数
        print("%d words missing from test data" % np.sum(test_sum == 0))

    sage_output = {'tr_data': train_X_sage, 'tr_aspect': tr_aspect, 'widx': tr_widx, 'vocab': vocab_for_sage}  # 存储训练词频，词表等
    if n_test > 0:
        sage_output['te_data'] = test_X_sage
        sage_output['te_aspect'] = te_aspect
    savemat(os.path.join(output_dir, 'sage_labeled.mat'), sage_output)
    sage_output['tr_aspect'] = tr_no_aspect
    if n_test > 0:
        sage_output['te_aspect'] = te_no_aspect
    savemat(os.path.join(output_dir, 'sage_unlabeled.mat'), sage_output)

    print("Done!")


def process_subset(items, parsed, label_fields, label_lists, vocab, output_dir, output_prefix):
    """
    对训练数据来说：
    items=所有的训练数据信息，按行划分
    parsed=训练文本处理后词的列表
    label_fields, label_lists = null
    vocab=全部文档的所有单词组成的词表
    output_dir=输出的文件夹
    output_prefix=输出的文件名
    """
    n_items = len(items)
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))  # 为词表vocab以字典的形式建立索引 [a:0, b:1 ...]

    ids = []
    for i, item in enumerate(items):
        if 'id' in item:
            ids.append(item['id'])  # 读取id索引
    if len(ids) != n_items:  # 如果不相同，就重新索引
        ids = [str(i) for i in range(n_items)]

    # create a label index using string representations
    for label_field in label_fields:
        label_list = label_lists[label_field]
        n_labels = len(label_list)
        label_list_strings = [str(label) for label in label_list]
        label_index = dict(zip(label_list_strings, range(n_labels)))

        # convert labels to a data frame
        if n_labels > 0:
            label_matrix = np.zeros([n_items, n_labels], dtype=int)
            label_vector = np.zeros(n_items, dtype=int)

            for i, item in enumerate(items):
                label = item[label_field]
                label_matrix[i, label_index[str(label)]] = 1
                label_vector[i] = label_index[str(label)]

            labels_df = pd.DataFrame(label_matrix, index=ids, columns=label_list_strings)
            labels_df.to_csv(os.path.join(output_dir, output_prefix + '.' + label_field + '.csv'))
            label_vector_df = pd.DataFrame(label_vector, index=ids, columns=[label_field])
            if n_labels == 2:
                label_vector_df.to_csv(os.path.join(output_dir, output_prefix + '.' + label_field + '_vector.csv'))

    rows = []  # 行
    cols = []  # 列
    vals = []  # 值

    dat_strings = []
    dat_labels = []
    mallet_strings = []
    fast_text_lines = []

    counter = Counter()
    word_counter = Counter()
    doc_lines = []
    print("Converting to count representations")
    for i, words in enumerate(parsed):  # 每一行都是文档的token
        # get the vocab indices of words that are in the vocabulary
        indices = [vocab_index[word] for word in words if word in vocab_index]  # 如果该词在此表中， 得到他的词表索引（把token转化成词表索引）
        word_subset = [word for word in words if word in vocab_index]  # 词表索引所对应的token内容（与indics一一对应）

        counter.clear()
        counter.update(indices)  # 计算在本文档中，该token的索引出现了几次
        word_counter.clear()
        word_counter.update(word_subset)  # 计算在本文档中，token出现了几次

        if len(counter.keys()) > 0:  # 存在词的话
            # udpate the counts
            mallet_strings.append(str(i) + '\t' + 'en' + '\t' + ' '.join(word_subset))  # '0    en  university college auditorium university university home private'

            dat_string = str(int(len(counter))) + ' '  # 出现了多少个不重复的词
            dat_string += ' '.join([str(k) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)  # '5 432:3 89:1 26:1 202:1 316:1' 出现了多少个不重复的词 出现的词的频率

            # for dat formart, assume just one label is given
            if len(label_fields) > 0:
                label = items[i][label_fields[-1]]
                dat_labels.append(str(label_index[str(label)]))

            values = list(counter.values())  # 对词频进行处理
            rows.extend([i] * len(counter))  # [0, 0, 0, 0, 0]  [1 * 700000]
            token_indices = sorted(counter.keys())  # 按照索引编号大小进行排序
            cols.extend(list(token_indices))  # [26, 89, 202, 316, 432] 每个文档的token索引
            vals.extend([counter[k] for k in token_indices])  # [1, 1, 1, 1, 3] 索引编号对应的值（该token出现了几次）

    # convert to a sparse representation
    sparse_X = sparse.coo_matrix((vals, (rows, cols)), shape=(n_items, vocab_size)).tocsr()  # 把每个文档的token以及出现的频率转换为矩阵的形式
    fh.save_sparse(sparse_X, os.path.join(output_dir, output_prefix + '.npz'))  # 输出npz文件 700000 * 455 700000行中的每一行对应一个文档，文档中又有token

    print("Size of {:s} document-term matrix:".format(output_prefix), sparse_X.shape)

    fh.write_to_json(ids, os.path.join(output_dir, output_prefix + '.ids.json'))

    # save output for Mallet
    fh.write_list_to_text(mallet_strings, os.path.join(output_dir, output_prefix + '.mallet.txt'))

    # save output for David Blei's LDA/SLDA code
    fh.write_list_to_text(dat_strings, os.path.join(output_dir, output_prefix + '.data.dat'))
    if len(dat_labels) > 0:
        fh.write_list_to_text(dat_labels, os.path.join(output_dir, output_prefix + '.' + label_field + '.dat'))

    # save output for Jacob Eisenstein's SAGE code:
    #sparse_X_sage = sparse.csr_matrix(X, dtype=float)
    vocab_for_sage = np.zeros((vocab_size,), dtype=object)  # [455, 0]
    vocab_for_sage[:] = vocab

    # for SAGE, assume only a single label has been given
    if len(label_fields) > 0:
        # convert array to vector of labels for SAGE
        sage_aspect = np.argmax(np.array(labels_df.values, dtype=float), axis=1) + 1
    else:
        sage_aspect = np.ones([n_items, 1], dtype=float)
    sage_no_aspect = np.array([n_items, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    return sparse_X, sage_aspect, sage_no_aspect, widx, vocab_for_sage


def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)  # 清洁文本
    tokens = text.split()  # 单词列表

    if stopwords is not None:  # 要对停词做处理
        tokens = ['_' if t in stopwords else t for t in tokens]  # 如果单词包括停词，则处理成'_'

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:  # 清除token列表里的所有数字
        tokens = [t if alpha.match(t) else '_' for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]

    # drop short tokens
    if min_length > 0:  # 如果最小长度大于0 去除字母个数小于3的单词
        tokens = [t if len(t) >= min_length else '_' for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != '_']  # 把所有'_'剔除，只留下token列表
    counts.update(unigrams)  # 使用字典进行词频计数

    if vocab is not None:  # 词表为空
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams  #

    return tokens, counts  # 返回当前文本所有token的列表，词频


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main(sys.argv[1:])

