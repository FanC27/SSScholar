import argparse
import numpy as np
import file_handling as fh

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

