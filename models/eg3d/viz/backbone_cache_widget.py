# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import imgui
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class BackboneCacheWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.cache_backbone = True

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text('Cache Backbone')
            imgui.same_line(viz.label_w + viz.spacing * 4)
            _clicked, self.cache_backbone = imgui.checkbox('##backbonecache', self.cache_backbone)
            imgui.same_line(viz.label_w + viz.spacing * 10)
            imgui.text('Note that when enabled, you may be unable to view intermediate backbone weights below')

        viz.args.do_backbone_caching = self.cache_backbone

#----------------------------------------------------------------------------
