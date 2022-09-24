"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths

from torch_utils import misc

from training.triplane import TriPlaneGenerator
import dnnlib as dnnlib
from configs.eg3d_config import init_kwargs,rendering_kwargs
import time
from models.eg3d import legacy

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		# self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		self.opts.n_styles = 14
		# Define architecture

		self.encoder = self.set_encoder()
		
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed

		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder.cuda()

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location=torch.device(self.opts.rank))
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:

			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'], map_location=torch.device(self.opts.rank))
			# decoder_ckpt = torch.load(model_paths['eg3d_pth'], map_location=torch.device(self.opts.rank))

			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			
			# with dnnlib.util.open_url(model_paths['eg3d_ffhq']) as f:
			# 	self.decoder = legacy.load_network_pkl(f)['G_ema']
			with dnnlib.util.open_url(model_paths['eg3d_ffhq']) as f:
				temp_decoder = legacy.load_network_pkl(f)['G_ema']

			self.decoder = TriPlaneGenerator(*temp_decoder.init_args, **temp_decoder.init_kwargs).eval().requires_grad_(False)
			misc.copy_params_and_buffers(temp_decoder, self.decoder, require_all=True)
			self.decoder.neural_rendering_resolution = temp_decoder.neural_rendering_resolution
			self.decoder.rendering_kwargs = temp_decoder.rendering_kwargs

			self.decoder.requires_grad_(True)
				
			self.latent_avg = None
			print("Done!")

	def forward(self, x,y_cams = None, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, return_pose = False, alpha=None):

		if input_code:
			codes = x
		else:
			codes, camera_params = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0


		input_is_latent = not input_code

		with torch.cuda.amp.autocast(enabled=False):
			if y_cams:
				images = self.decoder.synthesis(codes, y_cams)['image']
			else:
				images = self.decoder.synthesis(codes, camera_params)['image']
			
		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, camera_params, codes
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
