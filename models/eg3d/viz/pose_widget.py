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

class PoseWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.pose       = dnnlib.EasyDict(yaw=0, pitch=0, anim=False, speed=0.25)
        self.pose_def   = dnnlib.EasyDict(self.pose)

        self.lookat_point_choice = 0
        self.lookat_point_option = ['auto',         'ffhq',            'shapenet',         'afhq',         'manual']
        self.lookat_point_labels = ['Auto Detect',  'FFHQ Default',    'Shapenet Default', 'AFHQ Default', 'Manual']
        self.lookat_point = (0.0, 0.0, 0.2)

    def drag(self, dx, dy):
        viz = self.viz
        self.pose.yaw   += -dx / viz.font_size * 3e-2
        self.pose.pitch += -dy / viz.font_size * 3e-2

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Pose')
            imgui.same_line(viz.label_w)
            yaw = self.pose.yaw
            pitch = self.pose.pitch
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_yaw, new_pitch) = imgui.input_float2('##pose', yaw, pitch, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
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

            # New line starts here
            imgui.text('LookAt Point')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                _clicked, self.lookat_point_choice = imgui.combo('', self.lookat_point_choice, self.lookat_point_labels)
            lookat_point = self.lookat_point_option[self.lookat_point_choice]
            if lookat_point == 'auto':
                self.lookat_point = None
            if lookat_point == 'ffhq':
                self.lookat_point = (0.0, 0.0, 0.2)
                changes_enabled=False
            if lookat_point == 'shapenet':
                self.lookat_point = (0.0, 0.0, 0.0)
                changes_enabled=False
            if lookat_point == 'afhq':
                self.lookat_point = (0.0, 0.0, 0.0)
                changes_enabled=False
            if lookat_point == 'manual':
                if self.lookat_point is None:
                    self.lookat_point = (0.0, 0.0, 0.0)
                changes_enabled=True
            if lookat_point != 'auto':
                imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
                with imgui_utils.item_width(viz.font_size * 16):
                    with imgui_utils.grayed_out(not changes_enabled):
                        _changed, self.lookat_point = imgui.input_float3('##lookat', *self.lookat_point, format='%.2f', flags=(imgui.INPUT_TEXT_READ_ONLY if not changes_enabled else 0))


        viz.args.yaw   = self.pose.yaw
        viz.args.pitch = self.pose.pitch

        viz.args.lookat_point = self.lookat_point

#----------------------------------------------------------------------------
