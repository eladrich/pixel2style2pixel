from abc import abstractmethod
import torchvision.transforms as transforms
from datasets import augmentations
from options.train_options import TaskConfig


class TransformsConfig(object):

	def __init__(self, cfg: TaskConfig):
		self.cfg = cfg

	@abstractmethod
	def get_transforms(self):
		pass


class EncodeTransforms(TransformsConfig):

	def __init__(self, cfg: TaskConfig):
		super(EncodeTransforms, self).__init__(cfg)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': None,
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class FrontalizationTransforms(TransformsConfig):

	def __init__(self, cfg: TaskConfig):
		super(FrontalizationTransforms, self).__init__(cfg)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict


class SketchToImageTransforms(TransformsConfig):

	def __init__(self, cfg):
		super(SketchToImageTransforms, self).__init__(cfg)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor()]),
		}
		return transforms_dict


class SegToImageTransforms(TransformsConfig):

	def __init__(self, cfg):
		super(SegToImageTransforms, self).__init__(cfg)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.cfg.label_nc),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.cfg.label_nc),
				transforms.ToTensor()])
		}
		return transforms_dict


class SuperResTransforms(TransformsConfig):

	def __init__(self, cfg):
		super(SuperResTransforms, self).__init__(cfg)

	def get_transforms(self):
		factors = self.cfg.resize_factors
		if factors is None:
			factors = [1,2,4,8,16,32]
		print("Performing down-sampling with factors: {}".format(factors))
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.BilinearResize(factors=factors),
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict
