import cog
import tempfile
from pathlib import Path
from argparse import Namespace
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp
import dlib
from scripts.align_all_parallel import align_face


class Predictor(cog.Predictor):
    def setup(self):
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        model_paths = {
            "ffhq_frontalize": "pretrained_models/psp_ffhq_frontalization.pt",
            "celebs_sketch_to_face": "pretrained_models/psp_celebs_sketch_to_face.pt",
            "celebs_super_resolution": "pretrained_models/psp_celebs_super_resolution.pt",
            "toonify": "pretrained_models/psp_ffhq_toonify.pt"
        }

        loaded_models = {}
        for key, value in model_paths.items():
            loaded_models[key] = torch.load(value, map_location='cpu')

        self.opts = {}
        for key, value in loaded_models.items():
            self.opts[key] = value['opts']

        for key in self.opts.keys():
            self.opts[key]['checkpoint_path'] = model_paths[key]
            if 'learn_in_w' not in self.opts[key]:
                self.opts[key]['learn_in_w'] = False
            if 'output_size' not in self.opts[key]:
                self.opts[key]['output_size'] = 1024

        # loading all models here at once will get killed
        # self.nets = {}
        # for key, value in self.opts.items():
        #     self.nets[key] = pSp(Namespace(**value))

        self.transforms = {}
        for key in model_paths.keys():
            if key in ['ffhq_frontalize', 'toonify']:
                self.transforms[key] = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            elif key == 'celebs_sketch_to_face':
                self.transforms[key] = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()])
            elif key == 'celebs_super_resolution':
                self.transforms[key] = transforms.Compose([
                    transforms.Resize((256, 256)),
                    augmentations.BilinearResize(factors=[16]),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    @cog.input("model", type=str,
               options=["celebs_sketch_to_face", "ffhq_frontalize", "celebs_super_resolution", "toonify"],
               help="choose model type")
    @cog.input("image", type=Path, help="input facial image")
    def predict(self, model, image):
        opts = self.opts[model]
        opts = Namespace(**opts)
        pprint.pprint(opts)

        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')

        original_image = Image.open(str(image))
        if opts.label_nc == 0:
            original_image = original_image.convert("RGB")
        else:
            original_image = original_image.convert("L")
        original_image.resize((self.opts[model]['output_size'], self.opts[model]['output_size']))

        # Align Image
        if model not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
            input_image = self.run_alignment(str(image))
        else:
            input_image = original_image

        img_transforms = self.transforms[model]
        transformed_image = img_transforms(input_image)

        if model in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
            latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        else:
            latent_mask = None

        with torch.no_grad():
            result_image = run_on_batch(transformed_image.unsqueeze(0), net, latent_mask)[0]
        input_vis_image = log_input_image(transformed_image, opts)
        output_image = tensor2im(result_image)

        if model == "celebs_super_resolution":
            res = np.concatenate([np.array(input_vis_image.resize((self.opts[model]['output_size'], self.opts[model]['output_size']))),
                                  np.array(output_image.resize((self.opts[model]['output_size'], self.opts[model]['output_size'])))], axis=1)
        else:
            res = np.array(output_image.resize((self.opts[model]['output_size'], self.opts[model]['output_size'])))

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        Image.fromarray(np.array(res)).save(str(out_path))
        return out_path

    def run_alignment(self, image_path):
        aligned_image = align_face(filepath=image_path, predictor=self.predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image


def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
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
                      resize=False)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch
