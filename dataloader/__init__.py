from .dataloader_update import *
from .dataloader_neg import *
from .dataloader_pretrain import *
from .dataloader_neg_prebatch import *
from .gpt2_dataloader import *
from .knnlm_dataloader import *

def load_dataset(args):
    if args['mode'] in ['train', 'pretrain', 'train_neg']:
        dataset_name = args['models'][args['model']]['dataset_name'][args['mode']]
        dataset_t = globals()[dataset_name]
    else:
        raise Exception(f'[!] Unknown mode: {args["mode"]}')

    data = dataset_t(**args)

    if args['mode'] in ['train', 'pretrain', 'train_neg']:
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    else:
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        sampler = None
    return data, iter_, sampler
