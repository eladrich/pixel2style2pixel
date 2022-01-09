"""
This file runs the main training/val loop
"""
import sys

import pyrallis

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainConfig
from training.coach import Coach

@pyrallis.wrap()
def main(cfg: TrainConfig):
	if cfg.log.exp_dir.exists():
		raise Exception('Oops... {} already exists'.format(cfg.log.exp_dir))
	cfg.log.exp_dir.mkdir(parents=True)

	print(pyrallis.dump(cfg))
	with (cfg.log.exp_dir / 'config.yaml').open('w') as f:
		pyrallis.dump(cfg, f)

	coach = Coach(cfg)
	coach.train()


if __name__ == '__main__':
	main()
