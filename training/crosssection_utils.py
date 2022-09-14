# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

def sample_cross_section(G, ws, resolution=256, w=1.2):
    axis=0
    A, B = torch.meshgrid(torch.linspace(w/2, -w/2, resolution, device=ws.device), torch.linspace(-w/2, w/2, resolution, device=ws.device), indexing='ij')
    A, B = A.reshape(-1, 1), B.reshape(-1, 1)
    C = torch.zeros_like(A)
    coordinates = [A, B]
    coordinates.insert(axis, C)
    coordinates = torch.cat(coordinates, dim=-1).expand(ws.shape[0], -1, -1)

    sigma = G.sample_mixed(coordinates, torch.randn_like(coordinates), ws)['sigma']
    return sigma.reshape(-1, 1, resolution, resolution)

# if __name__ == '__main__':
#     sample_crossection(None)