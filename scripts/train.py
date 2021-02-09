"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
	opts = TrainOptions().parse()
	if os.path.exists(opts.exp_dir):
		if len(os.listdir(opts.exp_dir)) > 1:
			ans = input('Oops... {} already exists. Do you wish to continue training? [yes/no] '.format(opts.exp_dir))
			if ans == 'no':
				raise Exception('stop training! Please change exp_dir argument.'.format(opts.exp_dir))

	else:
		os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
