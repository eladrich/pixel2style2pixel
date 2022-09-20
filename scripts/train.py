"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import torch
import torch.multiprocessing as mp
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.eg3d_coach import Coach

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '25454'

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()

def main():

	# torch.autograd.set_detect_anomaly(True)
	opts = TrainOptions().parse()
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir,exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	if opts.distributed:
		mp.spawn(main_worker, nprocs = opts.num_gpus, args = (opts.num_gpus, opts), join = True)
	else:
		main_worker(0, opts.num_gpus, opts)

	
	

def main_worker(gpu, world_size, opts):
	print(gpu, world_size)
	opts.rank = gpu

	if opts.rank is not None:
		print("Use GPU: {} for training".format(opts.rank))
	
	setup(opts.rank, world_size)

	coach = Coach(opts)
	coach.train()

	cleanup()

if __name__ == '__main__':
	main()
