# Copyright 2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Viewer application module."""

import collections

from dm_control import _render
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import user_input
from dm_control.viewer import util
from dm_control.viewer import viewer
from dm_control.viewer import views
from dm_control.mujoco.wrapper import mjbindings
import glfw

mjlib = mjbindings.mjlib
_DOUBLE_BUFFERING = (user_input.KEY_F5)
_PAUSE = user_input.KEY_SPACE
_RESTART = user_input.KEY_BACKSPACE
_ADVANCE_SIMULATION = user_input.KEY_RIGHT
_SPEED_UP_TIME = user_input.KEY_EQUAL
_SLOW_DOWN_TIME = user_input.KEY_MINUS
_HELP = user_input.KEY_F1
_STATUS = user_input.KEY_F2

_MAX_FRONTBUFFER_SIZE = 2048
_MISSING_STATUS_ENTRY = '--'
_RUNTIME_STOPPED_LABEL = 'EPISODE TERMINATED - hit backspace to restart'
_STATUS_LABEL = 'Status'
_TIME_LABEL = 'Time'
_CPU_LABEL = 'CPU'
_FPS_LABEL = 'FPS'
_CAMERA_LABEL = 'Camera'
_PAUSED_LABEL = 'Paused'
_ERROR_LABEL = 'Error'

# Lower bound of the time multiplier set through TimeMultiplier class.
_MIN_TIME_MULTIPLIER = 1. / 32.
# Upper bound of the time multiplier set through TimeMultiplier class.
_MAX_TIME_MULTIPLIER = 2.


class TimeMultiplier:
    """Controls the relative speed of the simulation compared to realtime."""

    def __init__(self, initial_time_multiplier):
        """Instance initializer.

        Args:
          initial_time_multiplier: A float scalar specifying the initial speed of
            the simulation with 1.0 corresponding to realtime.
        """
        self.set(initial_time_multiplier)

    def get(self):
        """Returns the current time factor value."""
        return self._real_time_multiplier

    def set(self, value):
        """Modifies the time factor.

        Args:
          value: A float scalar, new value of the time factor.
        """
        self._real_time_multiplier = max(
            _MIN_TIME_MULTIPLIER, min(_MAX_TIME_MULTIPLIER, value))

    def __str__(self):
        """Returns a formatted string containing the time factor."""
        if self._real_time_multiplier >= 1.0:
            time_factor = '%d' % self._real_time_multiplier
        else:
            time_factor = '1/%d' % (1.0 // self._real_time_multiplier)
        return time_factor

    def increase(self):
        """Doubles the current time factor value."""
        self.set(self._real_time_multiplier * 2.)

    def decrease(self):
        """Halves the current time factor value."""
        self.set(self._real_time_multiplier / 2.)


class Help(views.ColumnTextModel):
    """Contains the description of input map employed in the application."""

    def __init__(self):
        """Instance initializer."""
        self._value = [
            ['Help', 'F1'],
            ['Info', 'F2'],
            ['Stereo', 'F5'],
            ['Frame', 'F6'],
            ['Label', 'F7'],
            ['--------------', ''],
            ['Pause', 'Space'],
            ['Reset', 'BackSpace'],
            ['Autoscale', 'Ctrl A'],
            ['Geoms', '0 - 4'],
            ['Sites', 'Shift 0 - 4'],
            ['Speed Up', '='],
            ['Slow Down', '-'],
            ['Switch Cam', '[ ]'],
            ['--------------', ''],
            ['Translate', 'R drag'],
            ['Rotate', 'L drag'],
            ['Zoom', 'Scroll'],
            ['Select', 'L dblclick'],
            ['Center', 'R dblclick'],
            ['Track', 'Ctrl R dblclick / Esc'],
            ['Perturb', 'Ctrl [Shift] L/R drag'],
        ]

    def get_columns(self):
        """Returns the text to display in two columns."""
        return self._value


class Status(views.ColumnTextModel):
    """Monitors and returns the status of the application."""

    def __init__(self, time_multiplier, pause, frame_timer):
        """Instance initializer.

        Args:
          time_multiplier: Instance of util.TimeMultiplier.
          pause: An observable pause subject, instance of util.ObservableFlag.
          frame_timer: A Timer instance counting duration of frames.
        """
        self.application = None
        self._time_multiplier = time_multiplier
        self._camera = None
        self._pause = pause
        self._frame_timer = frame_timer
        self._fps_counter = util.Integrator()
        self._cpu_counter = util.Integrator()

        self._value = collections.OrderedDict([
            (_STATUS_LABEL, _MISSING_STATUS_ENTRY),
            (_TIME_LABEL, _MISSING_STATUS_ENTRY),
            (_CPU_LABEL, _MISSING_STATUS_ENTRY),
            (_FPS_LABEL, _MISSING_STATUS_ENTRY),
            (_CAMERA_LABEL, _MISSING_STATUS_ENTRY),
            (_PAUSED_LABEL, _MISSING_STATUS_ENTRY),
            (_ERROR_LABEL, _MISSING_STATUS_ENTRY),
        ])

    def set_camera(self, camera):
        """Updates the active camera instance.

        Args:
          camera: Instance of renderer.SceneCamera.
        """
        self._camera = camera

    def set_application(self, application):
        self.application = application

    def get_columns(self):
        """Returns the text to display in two columns."""
        if self._frame_timer.measured_time > 0:
            self._fps_counter.value = 1. / self._frame_timer.measured_time
        self._value[_FPS_LABEL] = '{0:.1f}'.format(self._fps_counter.value)

        if self.application:
            self._cpu_counter.value = self.application._simulation_timer.measured_time
            self._value[_STATUS_LABEL] = "RUNNING"
            self._cpu_counter.value = 0

            self._value[_TIME_LABEL] = '{0:.1f} ({1}x)'.format(
                self.application.physics.data.time, str(self._time_multiplier))
            self._value[_CPU_LABEL] = '{0:.2f}ms'.format(
                self._cpu_counter.value * 1000.0)
        else:
            self._value[_STATUS_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_TIME_LABEL] = _MISSING_STATUS_ENTRY
            self._value[_CPU_LABEL] = _MISSING_STATUS_ENTRY

        if self._camera:
            self._value[_CAMERA_LABEL] = self._camera.name
        else:
            self._value[_CAMERA_LABEL] = _MISSING_STATUS_ENTRY

        self._value[_PAUSED_LABEL] = str(self._pause.value)

        return list(self._value.items())  # For Python 2/3 compatibility.

    def _clear_error(self):
        self._value[_ERROR_LABEL] = _MISSING_STATUS_ENTRY

    def _on_error(self, error_msg):
        self._value[_ERROR_LABEL] = error_msg


class ReloadParams(collections.namedtuple(
    'RefreshParams', ['zoom_to_scene'])):
    """Parameters of a reload request."""


class DMviewer:
    """Viewer application."""

    def __init__(self, physics, title='Explorer', width=1024, height=768):
        """Instance initializer."""
        self._render_surface = None
        self._renderer = renderer.NullRenderer()
        self._viewport = renderer.Viewport(width, height)
        self._window = gui.RenderWindow(width, height, title)

        self._pause_subject = util.ObservableFlag(True)
        self._time_multiplier = TimeMultiplier(1.)
        self._frame_timer = util.Timer()
        self._viewer = viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard)
        self._viewer_layout = views.ViewportLayout()
        self._status = Status(
            self._time_multiplier, self._pause_subject, self._frame_timer)

        self.physics = physics
        self.simulation_time_budget = 1. / 5.
        self.start = False
        self.clock = False
        self.end_time = 0
        self._simulation_timer = util.Timer()
        self._deferred_reload_request = None

        status_view_toggle = self._build_view_toggle(
            views.ColumnTextView(self._status), views.PanelLocation.BOTTOM_LEFT)
        help_view_toggle = self._build_view_toggle(
            views.ColumnTextView(Help()), views.PanelLocation.TOP_RIGHT)
        status_view_toggle()

        self._input_map = user_input.InputMap(
            self._window.mouse, self._window.keyboard)
        self._input_map.bind(self._pause_subject.toggle, _PAUSE)
        self._input_map.bind(self._time_multiplier.increase, _SPEED_UP_TIME)
        self._input_map.bind(self._time_multiplier.decrease, _SLOW_DOWN_TIME)
        # self._input_map.bind(self._advance_simulation, _ADVANCE_SIMULATION)
        # self._input_map.bind(self._restart_runtime, _RESTART)
        self._input_map.bind(help_view_toggle, _HELP)
        self._input_map.bind(status_view_toggle, _STATUS)
        self._on_reload()
        self._status.set_application(self)

    def reset(self):
        self.start = False
        self.clock = False
        self.end_time = 0

    def _on_reload(self, zoom_to_scene=False):
        """Perform initialization related to Physics reload.

        Reset the components that depend on a specific Physics class instance.

        Args:
          zoom_to_scene: Should the camera zoom to show the entire scene after the
            reload is complete.
        """
        self._deferred_reload_request = ReloadParams(zoom_to_scene)
        self._viewer.deinitialize()
        self._status.set_camera(None)

    def _perform_deferred_reload(self, params):
        """Performs the deferred part of initialization related to Physics reload.

        Args:
          params: Deferred reload parameters, an instance of ReloadParams.
        """
        if self._render_surface:
            self._render_surface.free()
        if self._renderer:
            self._renderer.release()
        self._render_surface = _render.Renderer(
            max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
        self._renderer = renderer.OffScreenRenderer(
            self.physics.model, self._render_surface)
        self._renderer.components += self._viewer_layout
        self._viewer.initialize(
            self.physics, self._renderer, touchpad=False)
        self._status.set_camera(self._viewer.camera)
        if params.zoom_to_scene:
            self._viewer.zoom_to_scene()

    def _build_view_toggle(self, view, location):
        def toggle():
            if view in self._viewer_layout:
                self._viewer_layout.remove(view)
            else:
                self._viewer_layout.add(view, location)

        return toggle

    def _tick(self):
        """Handle GUI events until the main window is closed."""
        if self._deferred_reload_request:
            self._perform_deferred_reload(self._deferred_reload_request)
            self._deferred_reload_request = None
        time_elapsed = self._frame_timer.tick() * self._time_multiplier.get()
        step_duration = min(time_elapsed, self.simulation_time_budget)
        with self._viewer.perturbation.apply(self._pause_subject.value):
            if self._pause_subject.value:
                self._step_paused()
            else:
                if not self.clock:
                    actual_simulation_time = self.get_time()
                    if self._tracked_simulation_time >= actual_simulation_time:
                        self.end_time = actual_simulation_time + step_duration
                        self.clock = True
                else:
                    if self.end_time <= self.get_time():
                        self.clock = False
                self._tracked_simulation_time += step_duration
        if not self.clock:
            self._viewer.render()

    def _step_paused(self):
        mjlib.mj_forward(self.physics.model.ptr, self.physics.data.ptr)

    def _render(self):
        if not glfw.window_should_close(self._window._context.window):
            self._tick()
            if not self.clock:
                self._viewport.set_size(*self._window.shape)
                pixels = self._renderer.pixels
                with self._window._context.make_current() as ctx:
                    ctx.call(
                        self._window._update_gui_on_render_thread, self._window._context.window, pixels)
                self._window._mouse.process_events()
                self._window._keyboard.process_events()
        else:
            self._window.close()
            raise WindowsError()

    def render(self):
        self._simulation_timer.tick()
        if not self.start:
            self._tracked_simulation_time = self.get_time()
            self.start = True
        if self._pause_subject.value:
            while self._pause_subject.value:
                self._render()
        else:
            self._render()

    def get_time(self):
        """Elapsed simulation time."""
        return self.physics.data.time