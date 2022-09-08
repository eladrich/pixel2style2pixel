# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import click
import os

import multiprocessing
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from viz import renderer
from viz import pickle_widget
from viz import latent_widget
from viz import stylemix_widget
from viz import trunc_noise_widget
from viz import performance_widget
from viz import capture_widget
from viz import backbone_cache_widget
from viz import layer_widget
from viz import pose_widget
from viz import zoom_widget
from viz import conditioning_pose_widget
from viz import render_type_widget
from viz import render_depth_sample_widget

#----------------------------------------------------------------------------

class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title='Cat Machine', window_width=3840, window_height=2160)

        # Internals.
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None

        # Widget interface.
        self.args               = dnnlib.EasyDict()
        self.result             = dnnlib.EasyDict()
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0

        # Widgets.
        self.pickle_widget      = pickle_widget.PickleWidget(self)
        self.latent_widget      = latent_widget.LatentWidget(self)
        self.stylemix_widget    = stylemix_widget.StyleMixingWidget(self)
        self.trunc_noise_widget = trunc_noise_widget.TruncationNoiseWidget(self)
        self.perf_widget        = performance_widget.PerformanceWidget(self)
        self.capture_widget     = capture_widget.CaptureWidget(self)
        self.backbone_cache_widget     = backbone_cache_widget.BackboneCacheWidget(self)
        self.layer_widget       = layer_widget.LayerWidget(self)
        self.pose_widget        = pose_widget.PoseWidget(self)
        self.zoom_widget        = zoom_widget.ZoomWidget(self)
        self.conditioning_pose_widget        = conditioning_pose_widget.ConditioningPoseWidget(self)
        self.render_type_widget = render_type_widget.RenderTypeWidget(self)
        self.render_depth_sample_widget = render_depth_sample_widget.RenderDepthSampleWidget(self)

        if capture_dir is not None:
            self.capture_widget.path = capture_dir

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def add_recent_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.add_recent(pkl, ignore_errors=ignore_errors)

    def load_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.load(pkl, ignore_errors=ignore_errors)

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_async(self, is_async):
        if is_async != self._async_renderer.is_async:
            self._async_renderer.set_async(is_async)
            self.clear_result()
            if 'image' in self.result:
                self.result.message = 'Switching rendering process...'
                self.defer_rendering()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def draw_frame(self):
        self.begin_frame()
        self.args = dnnlib.EasyDict()
        self.pane_w = self.font_size * 50
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 5.5)

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=self.content_width-self.pane_w, height=self.content_height)
        if dragging:
            self.pose_widget.drag(dx, dy)

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Network & latent', default=True)
        self.pickle_widget(expanded)
        self.pose_widget(expanded)
        self.zoom_widget(expanded)
        self.conditioning_pose_widget(expanded)
        self.render_type_widget(expanded)
        self.render_depth_sample_widget(expanded)
        self.latent_widget(expanded)
        self.stylemix_widget(expanded)
        self.trunc_noise_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Performance & capture', default=True)
        self.perf_widget(expanded)
        self.capture_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Layers & channels', default=True)
        self.backbone_cache_widget(expanded)
        self.layer_widget(expanded)

        # Render.
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result

        # Display.
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            # print(zoom)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            # zoom = 1
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

#----------------------------------------------------------------------------

class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        if args != self._cur_args:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer()
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp

#----------------------------------------------------------------------------

@click.command()
@click.argument('pkls', metavar='PATH', nargs=-1)
@click.option('--capture-dir', help='Where to save screenshot captures', metavar='PATH', default=None)
@click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
def main(
    pkls,
    capture_dir,
    browse_dir
):
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    viz = Visualizer(capture_dir=capture_dir)

    if browse_dir is not None:
        viz.pickle_widget.search_dirs = [browse_dir]

    # List pickles.
    pretrained = [
        'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhq512-128.pkl',
        'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/afhqcats512-128.pkl',
        'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-64.pkl',
        'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhqrebalanced512-128.pkl',
        'https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/shapenetcars128-64.pkl',
    ]

    # Populate recent pickles list with pretrained model URLs.
    for url in pretrained:
        viz.add_recent_pickle(url)

    # Run.
    while not viz.should_close():
        viz.draw_frame()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
