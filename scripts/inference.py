import os
import sys
import time

import numpy as np
import pyrallis
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import InferenceConfig
from options.train_options import TrainConfig
from models.psp import pSp
from utils import data_utils


@pyrallis.wrap()
def run(test_cfg: InferenceConfig):
    print('SCRIPT CONFIG:')
    print(pyrallis.dump(test_cfg))

    out_path_results = test_cfg.exp_dir / 'inference_results'
    out_path_coupled = test_cfg.exp_dir / 'inference_coupled'
    if test_cfg.resize_factor is not None:
        out_path_results /= f'downsampling_{test_cfg.resize_factor}'
        out_path_coupled /= f'downsampling_{test_cfg.resize_factor}'

    out_path_results.mkdir(parents=True, exist_ok=True)
    out_path_coupled.mkdir(parents=True, exist_ok=True)

    # Load model options
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

    if test_cfg.n_images is None:
        test_cfg.n_images = len(dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= test_cfg.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, test_cfg)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(test_cfg.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]

            if test_cfg.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], label_nc=model_cfg.task.label_nc)
                resize_amount = (256, 256) if test_cfg.resize_outputs else (
                model_cfg.task.output_size, model_cfg.task.output_size)
                if test_cfg.resize_factor is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(out_path_coupled / os.path.basename(im_path))

            im_save_path = out_path_results / os.path.basename(im_path)
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1

    stats_path = test_cfg.exp_dir / 'stats.txt'
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with stats_path.open('w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, cfg: InferenceConfig):
    if cfg.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=cfg.resize_outputs)
    else:
        latent_mask = cfg.latent_mask
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=cfg.mix_alpha,
                      resize=cfg.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
