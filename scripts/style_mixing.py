import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
	test_opts = TestOptions().parse()

	if test_opts.resize_factors is not None:
		factors = test_opts.resize_factors.split(',')
		assert len(factors) == 1, "When running inference, please provide a single downsampling factor!"
		mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing',
		                                  'downsampling_{}'.format(test_opts.resize_factors))
	else:
		mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing')
	os.makedirs(mixed_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	if 'output_size' not in opts:
		opts['output_size'] = 1024
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	print('Loading dataset for {}'.format(opts.dataset_type))
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path,
	                           transform=transforms_dict['transform_inference'],
	                           opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)

	latent_mask = [int(l) for l in opts.latent_mask.split(",")]
	if opts.n_images is None:
		opts.n_images = len(dataset)

	global_i = 0
	for input_batch in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		with torch.no_grad():
			input_batch = input_batch.cuda()
			for image_idx, input_image in enumerate(input_batch):
				# generate random vectors to inject into input image
				vecs_to_inject = np.random.randn(opts.n_outputs_to_generate, 512).astype('float32')
				multi_modal_outputs = []
				for vec_to_inject in vecs_to_inject:
					cur_vec = torch.from_numpy(vec_to_inject).unsqueeze(0).to("cuda")
					# get latent vector to inject into our input image
					_, latent_to_inject = net(cur_vec,
					                          input_code=True,
					                          return_latents=True)
					# get output image with injected style vector
					res = net(input_image.unsqueeze(0).to("cuda").float(),
					          latent_mask=latent_mask,
					          inject_latent=latent_to_inject,
					          alpha=opts.mix_alpha,
							  resize=opts.resize_outputs)
					multi_modal_outputs.append(res[0])

				# visualize multi modal outputs
				input_im_path = dataset.paths[global_i]
				image = input_batch[image_idx]
				input_image = log_input_image(image, opts)
				resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
				res = np.array(input_image.resize(resize_amount))
				for output in multi_modal_outputs:
					output = tensor2im(output)
					res = np.concatenate([res, np.array(output.resize(resize_amount))], axis=1)
				Image.fromarray(res).save(os.path.join(mixed_path_results, os.path.basename(input_im_path)))
				global_i += 1


if __name__ == '__main__':
	run()
