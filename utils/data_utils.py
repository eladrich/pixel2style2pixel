"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

import dataclasses
import pyrallis

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def decode_from_opts(config_class, opts_dict):
    """ A small function for converting existing flat opts to nested configs """
    nested_opts = {}
    if 'resize_factors' in opts_dict and opts_dict['resize_factors'] is not None:
        opts_dict['resize_factors'] = [int(val) for val in opts_dict['resize_factors'].split(',')]
    for config in dataclasses.fields(config_class):
        field_names = set([field.name for field in dataclasses.fields(config.type)])
        relevant_opts = {key:opts_dict[key] for key in opts_dict if key in field_names}
        nested_opts[config.name] = relevant_opts
    return pyrallis.decode(config_class, nested_opts)