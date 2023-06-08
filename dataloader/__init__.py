from .dataloader_update import *
from .dataloader_neg import *
from .dataloader_pretrain import *
from .dataloader_neg_prebatch import *
from .dataloader_queryside import *
from .dataloader_prebatch_all_ref import *
from .dataloader_prebatch_all_ref_all_candidate import *
from .dataloader_prebatch_all_ref_all_candidate_v2 import *
from .dataloader_prebatch_all_ref_all_candidate_v2_asyn import *
from .dataloader_prebatch_all_ref_all_candidate_v2_asyn_wikipedia import *
from .gpt2_dataloader import *
from .knnlm_dataloader import *

def load_dataset(args):
    dataset_name = args['models'][args['model']]['dataset_name'][args['dataset']][args['mode']]
    dataset_t = globals()[dataset_name]

    data = dataset_t(**args)
    if args['model'] == 'copyisallyouneed':
        sampler = torch.utils.data.SequentialSampler(data)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(data)

    iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)

    return data, iter_, sampler

def load_prebatch_processor(args):
    return PrebatchNegativePhrasesProcessor(**args)