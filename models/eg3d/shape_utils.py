# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
Utils for extracting 3D shapes using marching cubes. Based on code from DeepSDF (Park et al.)

Takes as input an .mrc file and extracts a mesh.

Ex.
    python shape_utils.py my_shape.mrc
Ex.
    python shape_utils.py myshapes_directory --level=12
"""


import time
import plyfile
import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import trimesh
import skimage.measure
import argparse
import mrcfile
from tqdm import tqdm
        

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")


def convert_mrc(input_filename, output_filename, isosurface_level=1):
    with mrcfile.open(input_filename) as mrc:
        convert_sdf_samples_to_ply(np.transpose(mrc.data, (2, 1, 0)), [0, 0, 0], 1, output_filename, level=isosurface_level)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mrc_path')
    parser.add_argument('--level', type=float, default=10, help="The isosurface level for marching cubes")
    args = parser.parse_args()

    if os.path.isfile(args.input_mrc_path) and args.input_mrc_path.split('.')[-1] == 'ply':
        output_obj_path = args.input_mrc_path.split('.mrc')[0] + '.ply'
        convert_mrc(args.input_mrc_path, output_obj_path, isosurface_level=1)

        print(f"{time.time() - start_time:02f} s")
    else:
        assert os.path.isdir(args.input_mrc_path)

        for mrc_path in tqdm(glob.glob(os.path.join(args.input_mrc_path, '*.mrc'))):
            output_obj_path = mrc_path.split('.mrc')[0] + '.ply'
            convert_mrc(mrc_path, output_obj_path, isosurface_level=args.level)