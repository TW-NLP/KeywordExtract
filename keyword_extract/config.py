import os
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

STOP_PATH = os.path.join(DATA_DIR, 'stop.txt')
PRE_MODEL = os.path.join(DATA_DIR, 'pre_model')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class KeyBERTConfig(object):
    BERT_MODEL = os.path.join(PRE_MODEL, 'bert_base_chinese')


class Word2vecConfig(object):
    # word2vec 参数
    W2V_VECTOR_SIZE = 128
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 1

    # DBSCAN 聚类参数
    DB_EPS = 0.01
    DB_MIN_SAMPLES = 2
    DB_METRIC = 'cosine'
