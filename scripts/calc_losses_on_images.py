from argparse import ArgumentParser
import os
import json
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from criteria.lpips.lpips import LPIPS
from datasets.gt_res_dataset import GTResDataset


def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--mode', type=str, default='lpips', choices=['lpips', 'l2'])
	parser.add_argument('--data_path', type=str, default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=4)
	args = parser.parse_args()
	return args


def run(args):

	transform = transforms.Compose([transforms.Resize((256, 256)),
									transforms.ToTensor(),
									transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	print('Loading dataset')
	dataset = GTResDataset(root_path=args.data_path,
	                       gt_dir=args.gt_path,
						   transform=transform)

	dataloader = DataLoader(dataset,
	                        batch_size=args.batch_size,
	                        shuffle=False,
	                        num_workers=int(args.workers),
	                        drop_last=True)

	if args.mode == 'lpips':
		loss_func = LPIPS(net_type='alex')
	elif args.mode == 'l2':
		loss_func = torch.nn.MSELoss()
	else:
		raise Exception('Not a valid mode!')
	loss_func.cuda()

	global_i = 0
	scores_dict = {}
	all_scores = []
	for result_batch, gt_batch in tqdm(dataloader):
		for i in range(args.batch_size):
			loss = float(loss_func(result_batch[i:i+1].cuda(), gt_batch[i:i+1].cuda()))
			all_scores.append(loss)
			im_path = dataset.pairs[global_i][0]
			scores_dict[os.path.basename(im_path)] = loss
			global_i += 1

	all_scores = list(scores_dict.values())
	mean = np.mean(all_scores)
	std = np.std(all_scores)
	result_str = 'Average loss is {:.2f}+-{:.2f}'.format(mean, std)
	print('Finished with ', args.data_path)
	print(result_str)

	out_path = os.path.join(os.path.dirname(args.data_path), 'inference_metrics')
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	with open(os.path.join(out_path, 'stat_{}.txt'.format(args.mode)), 'w') as f:
		f.write(result_str)
	with open(os.path.join(out_path, 'scores_{}.json'.format(args.mode)), 'w') as f:
		json.dump(scores_dict, f)


if __name__ == '__main__':
	args = parse_args()
	run(args)
