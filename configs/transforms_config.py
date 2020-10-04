from abc import abstractmethod
import torchvision.transforms as transforms
from datasets import augmentations


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass


class EncodeTransforms(TransformsConfig):

	def __init__(self, opts):
		super(EncodeTransforms, self).__init__(opts)

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

	def __init__(self, opts):
		super(FrontalizationTransforms, self).__init__(opts)

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

	def __init__(self, opts):
		super(SketchToImageTransforms, self).__init__(opts)

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

	def __init__(self, opts):
		super(SegToImageTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.opts.label_nc),
				transforms.ToTensor()]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				augmentations.ToOneHot(self.opts.label_nc),
				transforms.ToTensor()])
		}
		return transforms_dict


class SuperResTransforms(TransformsConfig):

	def __init__(self, opts):
		super(SuperResTransforms, self).__init__(opts)

	def get_transforms(self):
		if self.opts.resize_factors is None:
			self.opts.resize_factors = '1,2,4,8,16,32'
		factors = [int(f) for f in self.opts.resize_factors.split(",")]
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
