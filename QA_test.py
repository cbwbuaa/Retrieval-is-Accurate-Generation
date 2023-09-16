from header import *
from dataloader import *
from models import *
from config import *
import sys, collections, random
import numpy as np
import faiss
from time import time
sys.path.append('/apdcephfs/share_916081/ponybwcao/phrase_extraction/training')
from utils import *
import faiss.contrib.torch_utils
from copy import deepcopy
import json

DATA_PATH = {
    'ai2_arc_challenge': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/ai2_arc/data_challenge_test.jsonl',
    'ai2_arc_easy': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/ai2_arc/data_easy_test.jsonl',
    'copa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/copa/data_test.jsonl',
    'cosmos_qa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/cosmos_qa/data_validation.jsonl',
    'csqa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/csqa/data_validation.jsonl',
    'dream': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/dream/data_test.jsonl',
    'hellaswag': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/hellaswag/data_validation.jsonl',
    'mutual': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/mutual/data_validation.jsonl',
    'openbookqa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/openbookqa/data_main_test.jsonl',
    'piqa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/piqa/data_validation.jsonl',
    'race': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/race/data_all_test.jsonl',
    'sciq': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/sciq/data_test.jsonl',
    'story_cloze': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/story_cloze/data_test.jsonl',
    'truthfulqa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/truthful_qa/mc_data.jsonl',
    'winogrande': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/winogrande/data_debiased_validation.jsonl',
    'winograd_wsc': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/winograd_wsc/data_test.jsonl',
    'rte': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/rte/data_validation.jsonl',
    'boolq': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/boolq/data_validation.jsonl',
    'wizard': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/wizard/test_random_split.json',
    'swag': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/swag/validation.jsonl',
    'medmcqa': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/medmcqa/validation.jsonl',
    'medusmile': '/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/QA/medusmile/test.jsonl'

}

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--model', type=str, default='copyisallyouneed')
    parser.add_argument('--test_dataset', type=str, default='CoQA')
    return parser.parse_args()

def valid_choice(choice):
    return len(choice) >= 8 and 2 <= len(choice.split()) <= 10

def load_QA_dataset(dataset_name):
    data_path = DATA_PATH[dataset_name]
    dataset = []
    with open(data_path) as f:
        for line in f:
            line = json.loads(line)
            if dataset_name == 'ai2_arc_challenge':
                question = f'Question: {line["question"]} \n Answer:'
                choices = line['choices']['text']
                label = line['choices']['label'].index(line['answerKey'])
            elif dataset_name == 'ai2_arc_easy':
                question = f'Question: {line["question"]} \n Answer:'
                choices = line['choices']['text']
                label = line['choices']['label'].index(line['answerKey'])
            elif dataset_name == 'copa':
                question_map = {
                    'cause': 'because',
                    'effect': 'so'
                }
                question = f"{line['premise'].strip('.')} {question_map[line['question']]}"
                choices = [line['choice1'][0].lower() + line['choice1'][1:], line['choice2'][0].lower() + line['choice2'][1:]]
                label = line['label']
            elif dataset_name == 'cosmos_qa':
                question = line['context'] + ' ' + line['question']
                choices = [line['answer0'], line['answer1'], line['answer2'], line['answer3']]
                label = line['label']
            elif dataset_name == 'csqa':
                question = line['question']
                choices = line['choices']['text']
                label = line['choices']['label'].index(line['answerKey'])
            elif dataset_name == 'dream':
                question = ' '.join(line['dialogue']) + ' ' + line['question']
                choices = line['choice']
                label = line['choice'].index(line['answer'])
            elif dataset_name == 'hellaswag':
                question = f'{line["activity_label"]}: {line["ctx"]}'
                choices = line['endings']
                label = int(line['label'])
            elif dataset_name == 'mutual':
                question = line['article']
                choices = line['options']
                label = ['A', 'B', 'C', 'D'].index(line['answers'])
            elif dataset_name == 'openbookqa':
                question = line['question_stem']
                choices = line['choices']['text']
                label = line['choices']['label'].index(line['answerKey'])
            elif dataset_name == 'piqa':
                question = line['goal']
                choices = [line['sol1'], line['sol2']]
                label = line['label']
            elif dataset_name == 'race':
                question = f"{line['article']} \n Question: {line['question']} \n Answer:"
                choices = line['options']
                label = ['A', 'B', 'C', 'D'].index(line['answer'])
            elif dataset_name == 'sciq':
                question = line['question']
                choices = [line['correct_answer'], line['distractor1'], line['distractor2'], line['distractor3']]
                label = 0
            elif dataset_name == 'story_cloze':
                question = line['story']
                choices = line['choices']
                label = line['label'] - 1
            elif dataset_name == 'truthfulqa':
                question = line['question']
                choices = line['mc1_targets']['choices']
                label = line['mc1_targets']['labels'].index(1)
            elif dataset_name == 'winogrande':
                sentence = line['sentence']
                pos = sentence.find('_')
                question = sentence[:pos].strip()
                choices = [line['option1'] + sentence[pos + 1:], line['option2'] + sentence[pos + 1:]]
                label = int(line['answer']) - 1
            elif dataset_name == 'winograd_wsc':
                sentence = line['text']
                pos = line['pronoun_loc']
                question = sentence[:pos].strip()
                pronoun_len = len(line['pronoun'])
                choices = [x + sentence[pos + pronoun_len:] for x in line['options']]
                label = line['label']
            elif dataset_name == 'rte':
                question = f"{line['text1']} \n Question: {line['text2']} True or False? \n Answer:"
                choices = ['True', 'False']
                label = line['label']
            elif dataset_name == 'boolq':
                question = f"{line['passage']} \n Question: {line['question'].capitalize()}? \n Answer:"
                choices = ['False', 'True']
                label = int(line['answer'])
            elif dataset_name == 'swag':
                question = line['startphrase']
                choices = [line['ending0'], line['ending1'], line['ending2'], line['ending3']]
                label = line['label']
            elif dataset_name == 'medmcqa':
                question = line['question']
                choices = [line['opa'], line['opb'], line['opc'], line['opd']]
                label = line['cop']
            elif dataset_name == 'medusmile':
                question = line['question']
                choices = [line['options']['A'], line['options']['B'], line['options']['C'], line['options']['D']]
                label = ['A', 'B', 'C', 'D'].index(line['answer_idx'])
            else:
                raise NotImplementedError
            if all([not valid_choice(x) for x in choices]):
                continue
            item = {'question': question, 'choices': choices, 'label': label}
            dataset.append(item)
    return dataset

def load_wizard():
    path = DATA_PATH['wizard']
    dataset = []
    with open(path, 'r') as f:
        data = json.load(f)
        for item in data:
            dialog = item['dialog']
            if len(dialog) == 1:
                continue
            question = ' '.join([x['text'] for x in dialog[:-1]])
            choices = dialog[-1]['candidate_responses']
            article = []
            for passage in dialog[-1]['retrieved_passages']:
                for v in passage.values():
                    article.append(' '.join(v))
            label = choices.index(dialog[-1]['text'])
            item = {'article': article, 'question': question, 'choices': choices, 'label': label}
            dataset.append(item)
    return dataset

if __name__ == '__main__':
    args = vars(parser_args())

    # results = MRC_QA(**args)
    # with open(f'test_coqa.json', 'w') as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)


    # convert_to_entailment(sys.argv[1], sys.argv[2])

    # prepare_QA_data(**args)