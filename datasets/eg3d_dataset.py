from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torch
import json
import os

class EG3DDataset(Dataset):

	def __init__(self, dataset_path, opts, metadata = None, transform=None, is_train = True):
		self.dataset_path = dataset_path
		self.opts = opts

		with open(metadata, "r") as f:
			raw_metadata = json.load(f)

		self.metadata = [{'path': i[0], 'cams': i[1]} for i in raw_metadata['labels']]

		if is_train:
			self.metadata = self.metadata[:128000]
		else:
			self.metadata = self.metadata[128000:]
		self.transform = transform
		

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, index):
		img_path = os.path.join(self.dataset_path,self.metadata[index]['path'])
		from_im = Image.open(img_path)
		from_im = from_im.convert('RGB')

		if self.transform:
			from_im = self.transform(from_im)

		camera_param = torch.Tensor(self.metadata[index]['cams'])

		return from_im, camera_param
