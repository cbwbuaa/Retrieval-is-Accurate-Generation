from header import *
from dataloader import *
from models import *
from config import *
import sys, os, datetime
import torch.multiprocessing as mp
from time import time

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--mode', type=str, default='train_pipeline')
    parser.add_argument('--loss_type', type=str, default='focal_loss')
    parser.add_argument('--version', type=str)
    parser.add_argument('--pretrain_model_path', type=str)
    parser.add_argument('--training_data_dir', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--total_step', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--prebatch_step', type=int, default=5)
    parser.add_argument('--total_workers', type=int)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--random_initialize', type=bool, default=False)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--data_file_num', type=int, default=8)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--iter_to_accumulate', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=0)
    parser.add_argument('--prebatch_num', type=int, default=1000)
    parser.add_argument('--phrase_dim', type=int, default=256)
    return parser.parse_args()

# no prebatch
def data_proc(args, queue):
    if args['local_rank'] == 0:
        print('[!!] data process:', os.getpid())
    train_data, train_iter, sampler = load_dataset(args)
    while True:
        for x in train_iter:
            queue.put(x)

def asynchronous_load(args):
    queue = mp.Queue(50)
    data_generator = mp.Process(target=data_proc, args=(args, queue))
    data_generator.daemon = True
    data_generator.start()
    while True:
        batch = queue.get()
        yield batch
    # data_generator.join()

def main(**args):
    torch.cuda.empty_cache()
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=60*120))
    args['global_rank'] = dist.get_rank()
    print(f'[!] global rank: {args["global_rank"]}')

    mp.set_start_method('spawn')

    config = load_config(args)
    args.update(config)

    __stderr__ = sys.stderr
    sys.stderr = open(f'{args["log_dir"]}/{args["mode"]}/{args["version"]}.error', 'a')

    if args['local_rank'] == 0:
        # sum_writer = SummaryWriter(
        #     log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/{args["mode"]}/{args["version"]}',
        # )
        sum_writer = None
        print('log dir: ', args['log_dir'])
    else:
        sum_writer = None
        
    
    agent = load_model(args)
    # pbar = tqdm(total=args['total_step'])
    pbar = None
    current_step, over_train_flag = 0, False
    # sampler.set_epoch(0)    # shuffle for DDP
    # if agent.load_last_step:
    #     current_step = agent.load_last_step + 1
    #     if args['global_rank'] == 0:
    #         print(f'[!] load latest step: {current_step}')
    if args['local_rank'] == 0:
        print('[!] model process:', os.getpid())
    training_st_time = time()
    for batch in asynchronous_load(args):
        agent.train_model(
            batch, 
            recoder=sum_writer, 
            current_step=current_step, 
            pbar=pbar
        )
        current_step += 1
        if args['global_rank'] == 0 and current_step % args['save_every'] == 0 and current_step > 0:
            agent.evaluate_model(current_step)
            save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/{args["mode"]}/best_{args["version"]}_{current_step}.pt'
            agent.save_model_long(save_path, current_step)
            if args['local_rank'] == 0:
                sys.stderr.write(f'step: {current_step}, time: {(time() - training_st_time) // 60:.2f}\n')
        if current_step >= args['total_step']:
            # over_train_flag = True
            if args['local_rank'] == 0:
                sys.stderr.write('Finish.\n')
            break
        # if over_train_flag:
        #     break
        # if args['global_rank'] == 0:
        #     save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/{args["mode"]}/best_{args["version"]}_epoch{epoch}.pt'
        #     agent.save_model_long(save_path, current_step)
    if sum_writer:
        sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
