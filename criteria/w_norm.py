import torch
from torch import nn


class WNormLoss(nn.Module):

	def __init__(self, start_from_latent_avg=True):
		super(WNormLoss, self).__init__()
		self.start_from_latent_avg = start_from_latent_avg

	def forward(self, latent, latent_avg=None):
		if self.start_from_latent_avg:
			latent = latent - latent_avg
		return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
