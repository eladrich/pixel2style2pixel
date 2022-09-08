# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class ConditioningPoseWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.pose       = dnnlib.EasyDict(yaw=0, pitch=0, anim=False, speed=0.25)
        self.pose_def   = dnnlib.EasyDict(self.pose)

    def drag(self, dx, dy):
        viz = self.viz
        self.pose.yaw   += -dx / viz.font_size * 3e-2
        self.pose.pitch += -dy / viz.font_size * 3e-2

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Cond Pose')
            imgui.same_line(viz.label_w)
            yaw = self.pose.yaw
            pitch = self.pose.pitch
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_yaw, new_pitch) = imgui.input_float2('##frac', yaw, pitch, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.pose.yaw = new_yaw
                    self.pose.pitch = new_pitch
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            imgui.same_line()
            snapped = dnnlib.EasyDict(self.pose, yaw=round(self.pose.yaw, 1), pitch=round(self.pose.pitch, 1))
            if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.pose != snapped)):
                self.pose = snapped
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.pose != self.pose_def)):
                self.pose = dnnlib.EasyDict(self.pose_def)

        viz.args.conditioning_yaw   = self.pose.yaw
        viz.args.conditioning_pitch = self.pose.pitch

#----------------------------------------------------------------------------
