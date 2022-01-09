import os
import sys

import numpy as np
import pyrallis
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from options.train_options import TrainConfig
from utils import data_utils

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import MixingConfig
from models.psp import pSp


@pyrallis.wrap()
def run(test_cfg: MixingConfig):
    mixed_path_results = test_cfg.exp_dir / 'style_mixing'
    mixed_path_results.mkdir(parents=True, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_cfg.checkpoint_path, map_location='cpu')
    if 'cfg' in ckpt:
        model_cfg = pyrallis.decode(TrainConfig, ckpt['cfg'])
    elif 'opts' in ckpt:
        model_cfg = data_utils.decode_from_opts(TrainConfig, ckpt['opts'])
    else:
        raise Exception('No opts/cfg file!')

    print('MODEL CONFIG:')
    print(pyrallis.dump(model_cfg))

    model_cfg.task.checkpoint_path = test_cfg.checkpoint_path
    model_cfg.task.resize_factors = [test_cfg.resize_factor]

    net = pSp(model_cfg.task)

    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(model_cfg.task.dataset_type))
    dataset_args = data_configs.DATASETS[model_cfg.task.dataset_type]
    transforms_dict = dataset_args['transforms'](model_cfg.task).get_transforms()
    dataset = InferenceDataset(root=test_cfg.data_path,
                               transform=transforms_dict['transform_inference'],
                               cfg=model_cfg.task)
    dataloader = DataLoader(dataset,
                            batch_size=test_cfg.test_batch_size,
                            shuffle=False,
                            num_workers=int(test_cfg.test_workers),
                            drop_last=True)

    latent_mask = test_cfg.latent_mask
    if test_cfg.n_images is None:
        test_cfg.n_images = len(dataset)

    global_i = 0
    for input_batch in tqdm(dataloader):
        if global_i >= test_cfg.n_images:
            break
        with torch.no_grad():
            input_batch = input_batch.cuda()
            for image_idx, input_image in enumerate(input_batch):
                # generate random vectors to inject into input image
                vecs_to_inject = np.random.randn(test_cfg.n_outputs_to_generate, 512).astype('float32')
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
                              alpha=test_cfg.mix_alpha,
                              resize=test_cfg.resize_outputs)
                    multi_modal_outputs.append(res[0])

                # visualize multi modal outputs
                input_im_path = dataset.paths[global_i]
                image = input_batch[image_idx]
                input_image = log_input_image(image, label_nc=model_cfg.task.label_nc)
                resize_amount = (256, 256) if test_cfg.resize_outputs else (
                model_cfg.task.output_size, model_cfg.task.output_size)
                res = np.array(input_image.resize(resize_amount))
                for output in multi_modal_outputs:
                    output = tensor2im(output)
                    res = np.concatenate([res, np.array(output.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(mixed_path_results / os.path.basename(input_im_path))
                global_i += 1


if __name__ == '__main__':
    run()
