from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List




@dataclass
class BaseTestConfig:
	# Path to experiment output directory
	exp_dir: Path
	# Path to pSp model checkpoint
	checkpoint_path: Optional[str] = None
	# Path to directory of images to evaluate
	data_path: str = 'gt_images'
	# Whether to resize outputs to 256x256 or keep at 1024x1024
	resize_outputs: bool = False
	# Number of images to output. If None, run on all data
	n_images: Optional[int] = None
	# Batch size for testing and inference
	test_batch_size: int = 2
	# Number of test/inference dataloader workers
	test_workers: int = 2
	# List of latents to perform style-mixing with
	latent_mask: Optional[List[int]] = None
	# Alpha value for style-mixing
	mix_alpha: Optional[float] = None


@dataclass
class InferenceConfig(BaseTestConfig):
	""" arguments for inference script"""
	# Whether to also save inputs + outputs side-by-side
	couple_outputs: bool = False
	# Downsampling factor for super-res (should be a single value for inference).
	resize_factor: Optional[int] = None


@dataclass
class MixingConfig(BaseTestConfig):
	""" arguments for style-mixing script """
	# Number of outputs to generate per input image.
	n_outputs_to_generate: int = 5

