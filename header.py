import torch
from datasets import load_dataset
import inspect
from pynvml import *
from typing import List, Optional, Tuple, Union
from io import StringIO
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
import os
import sys
import re
import math
from itertools import chain
import csv
import jieba
from jieba import analyse
import jieba.posseg as pseg
import random
import json
import ijson
import time
import pprint
import hashlib
import logging
from copy import deepcopy
import ipdb
import transformers
from transformers import BertTokenizer, AutoModel, AutoConfig, AutoTokenizer, GPT2LMHeadModel
import pickle
from torch.cuda.amp import autocast, GradScaler
import argparse
from torch.nn.utils.rnn import pad_sequence
import joblib
from elasticsearch import Elasticsearch, helpers
import faiss
import torch.multiprocessing
import linecache
import nanopq
from scipy.stats import pearsonr, spearmanr
# from parallel.locker import MutexMap
# from parallel.owned_mutex import WithMutex
# from parallel.paralled_task import ParalledTask

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# def split_into_chunk(items, chunk_num):
#     datasets = []
#     size = len(items) // chunk_num
#     for chunk_idx in range(chunk_num):
#         st_idx = chunk_idx * size
#         end_idx = st_idx + size if chunk_idx != chunk_num - 1 else len(items)
#         datasets.append(items[st_idx: end_idx])
#     return datasets

# def check_false_neg_multiproc(suffix_list, phrase_list, nworker=8):
#     def worker(ctx: dict):
#         task = ctx.get('task')
#         bar: WithMutex = ctx.get('bar')
#         worker_id = ctx.get('worker_id')
#         suffix_idxs = ctx.get('datasets')[worker_id]

#         false_neg_idx = []
#         for suffix_idx in suffix_idxs:
#             suffix = suffix_list[suffix_idx]
#             if suffix:
#                 false_neg_idx_ = []
#                 for phrase_idx, phrase_ in enumerate(phrase_list):
#                     if suffix.startswith(phrase_):
#                         false_neg_idx_.append(phrase_idx)
#                 false_neg_idx.append((suffix_idx, false_neg_idx_))
#         return false_neg_idx
    
#     def reducer(results: list):
#         false_neg_mask = torch.ones(len(suffix_list), len(phrase_list))
#         for idx, result in results:
#             for suffix_idx, false_neg_idx_ in result:
#                 false_neg_mask[suffix_idx][false_neg_idx_] = 0
#         return false_neg_mask

#     datasets = split_into_chunk(list(range(len(suffix_list))), nworker)
#     false_neg_mask = ParalledTask.create('check false neg')\
#                 .set_nworker(nworker)\
#                 .set_worker_args({'datasets': datasets})\
#                 .set_worker_func(worker)\
#                 .set_reducer_func(reducer)\
#                 .execute()\
#                 .get_results()
#     return false_neg_mask
