# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from inspect import formatargvalues
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class ZoomWidget:
    def __init__(self, viz):
        self.viz         = viz
        self.fov         = 18.837
        self.fov_default = 18.837

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('FOV')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                _changed, self.fov = imgui.slider_float('##fov', self.fov, 12, 45, format='%.2f Degrees')

            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.button_w + viz.spacing * 3)
            snapped = round(self.fov)
            if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.fov != snapped)):
                self.fov = snapped
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(abs(self.fov - self.fov_default)) > .01):
                self.fov = self.fov_default

        viz.args.focal_length = float(1 / (np.tan(self.fov * 3.14159 / 360) * 1.414))
#----------------------------------------------------------------------------
