from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import math

from configs.paths_config import model_paths


@dataclass
class ComputeConfig:
	""" Arguments related to workers and batches """
	# Batch size for training
	batch_size: int = 4
	# Batch size for testing and inference
	test_batch_size: int = 2
	# Number of train dataloader workers
	workers: int = 4
	# Number of test/inference dataloader workers
	test_workers: int = 2


@dataclass
class OptimConfig:
	""" Arguments related to the optimization """
	# Optimizer learning rate
	learning_rate: float = 0.0001
	# Which optimizer to use
	optim_name: str = 'ranger'
	# Maximum number of training steps
	max_steps: int = 500000
	# Whether to train the decoder model
	train_decoder: bool = False



@dataclass
class LossConfig:
	""" Arguments related to the loss function """
	# LPIPS loss multiplier factor
	lpips_lambda: float = 0.8
	# ID loss multiplier factor
	id_lambda: float = 0
	# L2 loss multiplier factor
	l2_lambda: float = 1.0
	# W-norm loss multiplier factor
	w_norm_lambda: float = 0
	# LPIPS loss multiplier factor for inner image region
	lpips_lambda_crop: float = 0
	# L2 loss multiplier factor for inner image region
	l2_lambda_crop: float = 0
	# Moco-based feature similarity loss multiplier factor
	moco_lambda: float = 0


@dataclass
class LogConfig:
	""" Arguments related to logging """
	# Path to experiment output directory
	exp_dir: Path
	# Interval for logging train images during training
	image_interval: int = 100
	# Interval for logging metrics to tensorboard
	board_interval: int = 50
	# Validation interval
	val_interval: int = 1000
	# Model checkpoint interval
	save_interval: Optional[int] = None
	# Whether to use Weights & Biases to track experiment.
	use_wandb: bool = False


@dataclass
class TaskConfig:
	""" Arguments related to the model and task """
	# Type of dataset/experiment to run
	dataset_type: str = 'ffhq_encode'
	# Number of input image channels to the psp encoder
	input_nc: int = 3
	# Number of input label channels to the psp encoder
	label_nc: int = 0
	# Output size of generator
	output_size: int = 1024
	# For super-res, list of resize factors to use for inference.
	resize_factors: Optional[List[int]] = None
	# Which encoder to use
	encoder_type: str = 'GradualStyleEncoder'
	# Path to StyleGAN model weights
	stylegan_weights: str = model_paths['stylegan_ffhq']
	# Path to pSp model checkpoint
	checkpoint_path: str = None
	# Whether to add average latent vector to generate codes from encoder
	start_from_latent_avg: bool = False
	# Whether to learn in w space instead of w+
	learn_in_w: bool = False

	@property
	def n_styles(self) -> int:
		return int(math.log(self.output_size, 2)) * 2 - 2



@dataclass
class TrainConfig:
	""" All Training Arguments """
	compute: ComputeConfig = field(default_factory=ComputeConfig)
	optim: OptimConfig = field(default_factory=OptimConfig)
	loss: LossConfig = field(default_factory=LossConfig)
	log: LogConfig = field(default_factory=LogConfig)
	task: TaskConfig = field(default_factory=TaskConfig)
