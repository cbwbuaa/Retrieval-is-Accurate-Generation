from header import *
from dataloader import *
from models import *
from config import *
import sys, collections, random
from datasets import load_dataset
import numpy as np
import faiss
from time import time
sys.path.append('/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/training')
from tokenizer import Tokenizer
from utils import *
import faiss.contrib.torch_utils
from copy import deepcopy
import json

dataset_key = {
    'truthful_qa': 'validation',
}

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--model', type=str, default='copyisallyouneed')
    parser.add_argument('--model_name', type=str, default='best_0520_pipeline_update_lr1e-4_prebatch5_beta0.5_warmup50000_prenum200_temp1.5_400000')
    parser.add_argument('--test_dataset', type=str, default='CoQA')
    return parser.parse_args()

def MRC_QA(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/ckpt/wikitext103/copyisallyouneed/train_pipeline/{args["model_name"]}.pt')
    print(f'[!] init model over')

    dataset = load_dataset('coqa', 'train', cache_dir='/apdcephfs/share_916081/shared_info/ponybwcao/Copyisallyouneed/data/QA') # dict_keys(['train', 'validation'])
    print(f"[!] load {len(dataset['validation'])} examples.")
    results = []
    for x in tqdm(dataset['validation']): # dict_keys(['source', 'story', 'questions', 'answers'])
        story = x['story']
        questions_ = x['questions']
        questions = questions_
        # questions = [story + ' ' + x for x in questions_]
        answers = x['answers']['input_text']
        convert_question_to_statement(questions, answers)
        exit()
        preds = agent.test_MRCQA(story, questions, answers)
        # print(story)
        # for q, a, p in zip(questions_, answers, preds):
        #     print(f'q:{q}, a:{a}, p:{p}')
        # exit()
        results.append({
            'story': story,
            'questions': questions,
            'answers': answers,
            'preds': preds
        })
    return results

def truthful_QA():
    dataset = []
    with open('/apdcephfs/share_916081/shared_info/ponybwcao/phrase_extraction/data/QA/truthful_qa/mc_data_candidates_refs.jsonl') as f:
        for line in tqdm(f):
            dataset.append(json.loads(line))

    test_set = 'mc1_targets'
    collection = []
    doc_set = set()
    doc_list = []
    for x in tqdm(dataset):
        choices = x[test_set]['choices']
        choices_candidates = x[test_set]['candidates']
        choices_refs = x[test_set]['ref_docs']
        for refs in choices_refs:
            for ref in refs:
                if not ref:
                    continue
                doc_set |= set([x[0] for x in ref])
                doc_list += [x[0] for x in ref]
    # 1089884
    # 58928081
        

if __name__ == '__main__':
    args = vars(parser_args())

    # results = MRC_QA(**args)
    # with open(f'test_coqa.json', 'w') as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)


    # convert_to_entailment(sys.argv[1], sys.argv[2])

    # prepare_QA_data(**args)
    truthful_QA()