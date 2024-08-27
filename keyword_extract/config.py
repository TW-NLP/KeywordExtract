import os
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

STOP_PATH = os.path.join(DATA_DIR, 'stop.txt')
PRE_MODEL = os.path.join(DATA_DIR, 'pre_model')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class KeyBERTConfig(object):
    BERT_MODEL = os.path.join(PRE_MODEL, 'bert_base_chinese')
