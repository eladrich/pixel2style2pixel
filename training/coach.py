import os
import matplotlib
import matplotlib.pyplot as plt
import pyrallis

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from options.train_options import TrainConfig

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger


class Coach:
	def __init__(self, cfg: TrainConfig):
		self.cfg = cfg

		self.global_step = 0

		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES

		if self.cfg.log.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.cfg)

		# Initialize network
		self.net = pSp(self.cfg.task).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		if self.cfg.loss.id_lambda > 0 and self.cfg.loss.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.cfg.loss.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.cfg.loss.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.cfg.loss.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.cfg.task.start_from_latent_avg)
		if self.cfg.loss.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.cfg.compute.batch_size,
										   shuffle=True,
										   num_workers=int(self.cfg.compute.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.cfg.compute.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.cfg.compute.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = self.cfg.log.exp_dir / 'logs'
		log_dir.mkdir(parents=True, exist_ok=True)
		self.logger = SummaryWriter(log_dir=str(log_dir))

		# Initialize checkpoint dir
		self.checkpoint_dir = self.cfg.log.exp_dir / 'checkpoints'
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.best_val_loss = None
		if self.cfg.log.save_interval is None:
			self.cfg.log.save_interval = self.cfg.optim.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.cfg.optim.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()
				y_hat, latent = self.net.forward(x, return_latents=True)
				loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.cfg.log.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
				if self.global_step % self.cfg.log.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.cfg.log.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step, cfg=self.cfg)

				# Validation related
				val_loss_dict = None
				if self.global_step % self.cfg.log.val_interval == 0 or self.global_step == self.cfg.optim.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.cfg.log.save_interval == 0 or self.global_step == self.cfg.optim.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.cfg.optim.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch

			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				y_hat, latent = self.net.forward(x, return_latents=True)
				loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# Log images of first batch to wandb
			if self.cfg.log.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="test", step=self.global_step, cfg=self.cfg)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = self.checkpoint_dir / save_name
		torch.save(save_dict, checkpoint_path)
		with (self.checkpoint_dir / 'timestamp.txt').open('a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.cfg.log.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.cfg.optim.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.cfg.optim.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.cfg.optim.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.cfg.optim.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.cfg.task.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.cfg.task.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.cfg.task.dataset_type}')
		dataset_args = data_configs.DATASETS[self.cfg.task.dataset_type]
		transforms_dict = dataset_args['transforms'](self.cfg.task).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  label_nc=self.cfg.task.label_nc)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 label_nc=self.cfg.task.label_nc)
		if self.cfg.log.use_wandb:
			self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
			self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		loss_cfg = self.cfg.loss
		if loss_cfg.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * loss_cfg.id_lambda
		if loss_cfg.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * loss_cfg.l2_lambda
		if loss_cfg.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * loss_cfg.lpips_lambda
		if loss_cfg.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * loss_cfg.lpips_lambda_crop
		if loss_cfg.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * loss_cfg.l2_lambda_crop
		if loss_cfg.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * loss_cfg.w_norm_lambda
		if loss_cfg.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * loss_cfg.moco_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
		if self.cfg.log.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], label_nc=self.cfg.task.label_nc),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'cfg': pyrallis.encode(self.cfg)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.cfg.task.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
