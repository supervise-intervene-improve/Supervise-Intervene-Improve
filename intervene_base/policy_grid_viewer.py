import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import glfw
import mujoco
import numpy as np

# Desktop-RGB measurement reuses the VR runtime's collector rather than growing a second
# one. The point is comparability: `state_to_render_ms` has to mean the same thing in the
# desktop condition as in the two VR conditions, or the three numbers cannot go in one
# table. It also means tools/perf_report.py reads these rows unchanged.
_PERF_METRICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "SimPublisher", "sii", "integration_v1",
)
if _PERF_METRICS_DIR not in sys.path:
    sys.path.append(_PERF_METRICS_DIR)
try:
    from perf_metrics import MetricsCollector as _MetricsCollector
except Exception:  # pragma: no cover - measurement rig absent
    _MetricsCollector = None

from model_variants.descriptor import (
    BASE_KEY, is_valid_descriptor, scene_closure_sha256, variant_key,
)
from rendering.viewer import SceneViewer


DETAIL_CAMERAS = ("front", "left", "right", "wrist")
DETAIL_SIDE_ROLL_180_CAMERAS = {"left", "right"}
# Aspect the detail panes are letterboxed to, so each camera's whole image is visible
# rather than horizontally cropped to a portrait cell. 4:3 matches the scenes'
# <global offwidth/offheight> (640x480) and the VR panels. 0 disables the fit.
DETAIL_ASPECT = 4.0 / 3.0
PAUSED_LABEL = "|| PAUSED"


# Rows in the VR selector grid (MultiSessionGridManager.rows). The desktop grid mirrors
# it so a session occupies the SAME cell in both interfaces -- an operator switching
# between the desktop grid and the headset must not have to re-learn where S03 is, and a
# study log that says "the participant selected the middle-left cell" has to mean one
# thing. Only the POPULATED sub-grid is mirrored: VR always lays out against 5 columns and
# hides the empty panels, so at 9 windows its 3 filled columns are what the desktop draws.
VR_GRID_ROWS = 3
GRID_LAYOUT_VR = "vr"
GRID_LAYOUT_COMPACT = "compact"


def compute_grid(n: int, layout: str = GRID_LAYOUT_VR):
    """(rows, cols) for n tiles.

    'vr'      -- VR_GRID_ROWS rows, ceil(n/rows) columns: identical to the VR selector's
                 populated sub-grid at every supported window count (3/6/9/12/15).
    'compact' -- the legacy near-square sqrt layout. Kept because it packs an arbitrary
                 n more squarely; it does NOT agree with VR except by coincidence at 9
                 (both 3x3, but VR fills it column-major and this fills it row-major, so
                 even there every session except the diagonal sat in the wrong cell).
    """
    if layout == GRID_LAYOUT_VR:
        rows = max(1, min(VR_GRID_ROWS, n))
        cols = max(1, math.ceil(n / rows))
        return rows, cols
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def compute_viewports(win_w: int, win_h: int, n: int, pad: int = 8,
                      layout: str = GRID_LAYOUT_VR):
    return compute_viewports_in_rect(
        mujoco.MjrRect(0, 0, win_w, win_h), n, pad=pad, layout=layout
    )


def compute_viewports_in_rect(parent, n: int, pad: int = 8,
                              layout: str = GRID_LAYOUT_VR):
    """Rects in SESSION-INDEX order: rects[i] is where session i is drawn.

    The fill order is baked in here rather than applied as a permutation elsewhere, so
    rendering (_render_tile) and hit-testing (_pick_tile) cannot disagree about which
    cell a session owns -- they both just enumerate this list.
    """
    rows, cols = compute_grid(n, layout)
    cell_w = max(1, int(parent.width) // cols)
    cell_h = max(1, int(parent.height) // rows)
    parent_left = int(parent.left)
    parent_top = int(parent.bottom) + int(parent.height)

    rects = []
    for i in range(n):
        if layout == GRID_LAYOUT_VR:
            # Column-major, matching MultiSessionGridManager's columnMajorFill:
            # sessions 0,1,2 fill the LEFT column, 3,4,5 the next, etc.
            #
            # Bottom-up within each column. VR's row index is not a screen row: it
            # becomes `rowOffset = row - (rows-1)*0.5` and is applied as
            # `AngleAxis(-rowOffset * verticalAngleDeg, hRight)`. The negation means
            # row 0 pitches the panel DOWN, so VR row 0 is the BOTTOM of the column and
            # session 0 sits bottom-left. `r` here is a screen row measured from the
            # top, so it has to be inverted to agree.
            c, vr_row = divmod(i, rows)
            r = (rows - 1) - vr_row
        else:
            r, c = divmod(i, cols)

        x = parent_left + c * cell_w + pad
        y_top = r * cell_h + pad
        w = max(1, cell_w - 2 * pad)
        h = max(1, cell_h - 2 * pad)

        y = parent_top - (y_top + h)
        rects.append(mujoco.MjrRect(x, y, w, h))

    return rects


def compute_split_layout(win_w: int, win_h: int, gutter: int = 10):
    gutter = min(max(0, int(gutter)), max(0, int(win_w) - 2))
    left_w = max(1, (int(win_w) - gutter) // 2)
    right_w = max(1, int(win_w) - gutter - left_w)
    return (
        mujoco.MjrRect(0, 0, left_w, int(win_h)),
        mujoco.MjrRect(left_w + gutter, 0, right_w, int(win_h)),
    )


def compute_detail_camera_rects(parent, pad: int = 10, aspect: float = DETAIL_ASPECT):
    width = int(parent.width)
    height = int(parent.height)
    left = int(parent.left)
    bottom = int(parent.bottom)
    cell_w = max(1, width // 2)
    cell_h = max(1, height // 2)
    top_y = bottom + height - cell_h

    return {
        "front": _padded_rect(left, top_y, cell_w, cell_h, pad, aspect),
        "left": _padded_rect(left + cell_w, top_y, width - cell_w, cell_h, pad, aspect),
        "right": _padded_rect(left, bottom, cell_w, height - cell_h, pad, aspect),
        "wrist": _padded_rect(left + cell_w, bottom, width - cell_w, height - cell_h, pad, aspect),
    }


def _fit_aspect(left: int, bottom: int, width: int, height: int, aspect: float):
    """Shrink a rect to `aspect`, centred, so the camera's FULL view is visible.

    mjr_render keeps the camera's fovy and derives the HORIZONTAL field of view from the
    viewport aspect, so a rect narrower than the camera's own aspect silently crops the
    sides of the image rather than letterboxing it. The 2x2 detail cells are roughly
    portrait, which cut a visible slice off every pane. Letterbox instead: the cells are
    already cleared to the panel background, so the bars cost nothing.
    """
    if aspect is None or aspect <= 0.0:
        return int(left), int(bottom), max(1, int(width)), max(1, int(height))

    width = max(1, int(width))
    height = max(1, int(height))
    if width >= int(round(height * aspect)):
        fit_h = height
        fit_w = max(1, int(round(height * aspect)))
    else:
        fit_w = width
        fit_h = max(1, int(round(width / aspect)))
    return (
        int(left) + (width - fit_w) // 2,
        int(bottom) + (height - fit_h) // 2,
        fit_w,
        fit_h,
    )


def _padded_rect(left: int, bottom: int, width: int, height: int, pad: int,
                 aspect: float | None = None):
    pad = max(0, int(pad))
    left, bottom, width, height = _fit_aspect(
        int(left) + pad,
        int(bottom) + pad,
        max(1, int(width) - 2 * pad),
        max(1, int(height) - 2 * pad),
        aspect if aspect is not None else 0.0,
    )
    return mujoco.MjrRect(left, bottom, width, height)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def risk_color(risk: float) -> tuple[float, float, float, float]:
    """Render the raw ACC score gradient: green -> yellow -> red over 0..1."""
    try:
        value = float(risk)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        value = 0.0

    t = max(0.0, min(1.0, value))
    if t < 0.5:
        k = t * 2.0
        return (k, 1.0, 0.0, 1.0)

    k = (t - 0.5) * 2.0
    return (1.0, 1.0 - k, 0.0, 1.0)


@dataclass
class SessionState:
    index: int
    data: mujoco.MjData
    sub: object
    cmd: object
    latest: dict | None = None
    last_rx_wall: float = 0.0
    applied_seq: int | None = None
    # Publisher wall-clock of the state currently applied to `data`. Both processes run on
    # the same host, so this is directly comparable to time.time() here -- no clock sync,
    # and the same quantity the VR runtime calls `state_to_render_ms`.
    applied_state_wall: float = 0.0
    applied_states: int = 0
    stale_warned: bool = False
    episode_id: str | None = None
    # --- model variant ---------------------------------------------------------------
    # Which compiled model this session is publishing from. Sessions can differ: a cups OOD
    # episode runs a variant of the same scene with different cup heights and swapped box
    # textures. nq/nv/nu are identical across variants, so without tracking this the grid
    # would happily render the right numbers on the wrong geometry and never notice.
    variant_key: str = BASE_KEY
    variant_desc: dict | None = None
    render_key: str = BASE_KEY          # the slot actually used (may lag variant_key)
    data_key: str = BASE_KEY            # which slot's model `data` was built from
    degraded: bool = False              # rendering through the base slot, badged
    incompatible: bool = False          # different scene file entirely: refuse, loudly
    model_epoch: int = 0


@dataclass
class VariantSlot:
    """One compiled model plus the GL objects bound to it.

    `MjvScene` is allocated from the model and `MjrContext` uploads that model's meshes and
    textures, so both are per-model. Measured on the cups scene: ~404 MB RAM and ~137 MB
    VRAM per extra slot (101 MB of the VRAM is texture data).
    """
    key: str
    model: mujoco.MjModel
    viewer: SceneViewer
    ctx: object
    last_used: float = field(default_factory=time.time)


class PolicyGridViewer:
    def __init__(
        self,
        *,
        xml: str,
        sessions: int,
        state_host: str,
        state_base_port: int,
        cmd_host: str,
        cmd_base_port: int,
        port_step: int,
        width: int,
        height: int,
        cam: str,
        detail_cameras: tuple[str, str, str, str],
        detail_aspect: float,
        flip_front: bool,
        roll_side_cameras: bool,
        observe_only: bool,
        stale_timeout_s: float,
        metrics_out: str | None = None,
        metrics_window_s: float = 10.0,
        metrics_summary: bool = False,
        variant_render: str = "exact",
        variant_cache_size: int = 4,
        grid_layout: str = GRID_LAYOUT_VR,
    ):
        self.xml = xml
        self.sessions = int(sessions)
        self.state_host = state_host
        self.state_base_port = int(state_base_port)
        self.cmd_host = cmd_host
        self.cmd_base_port = int(cmd_base_port)
        self.port_step = int(port_step)
        self.width = int(width)
        self.height = int(height)
        self.cam = cam
        self.detail_cameras = detail_cameras
        self.detail_aspect = float(detail_aspect)
        self.flip_front = bool(flip_front)
        self.roll_side_cameras = bool(roll_side_cameras)
        self.observe_only = bool(observe_only)
        self.stale_timeout_s = float(stale_timeout_s)
        # Cell arrangement. Defaults to the VR selector's, so session N is in the same
        # place on the desktop as it is in the headset.
        self.grid_layout = (
            grid_layout if grid_layout in (GRID_LAYOUT_VR, GRID_LAYOUT_COMPACT)
            else GRID_LAYOUT_VR
        )
        # exact = compile a render slot per distinct variant on screen.
        # base  = render every session with the base model, badged. Zero extra memory; the
        #         policy sim and the VR mirror are still exact, only this monitor view is
        #         approximate.
        # off   = like base, without the badge (screenshots).
        self.variant_render = str(variant_render or "exact").strip().lower()
        self.variant_cache_size = max(1, int(variant_cache_size))
        self.slots: dict[str, VariantSlot] = {}
        self.variant_cache = None
        self._variant_warned: set = set()
        # Identity of OUR base geometry. A session reporting a different one was launched on
        # a different XML, which we refuse rather than render as if it matched.
        try:
            self._scene_closure_sha256 = scene_closure_sha256(self.xml)
        except Exception:
            self._scene_closure_sha256 = ""

        self.window = None
        self.ctx = None
        self.zmq_ctx = None
        self.model = None
        self.viewer = None
        self.session_states: list[SessionState] = []
        self.display_order: list[int] = []
        self.offscreen_width = max(self.width, 32)
        self.offscreen_height = max(self.height, 32)

        self.selected_idx = 0
        self.zoomed = False
        self.last_click_idx = None
        self.last_click_wall = 0.0
        self._observe_hint_printed = False
        # HRI block recording. The grid is the process that OWNS the desktop selection,
        # so it is the one that can timestamp a confirmed mouse click. Nothing else in
        # this viewer changes; it stays a read-only mirror when --observe_only.
        self._block_log = None
        self._block = None
        self._selection_logged = False

        # --- Desktop-RGB measurement ---------------------------------------------
        # This viewer had NO timing instrumentation at all: it is vsync-locked
        # (glfw.swap_interval(1) in init()) and consumes the 30 Hz state mirror, but
        # nothing observed either, so the Desktop-RGB condition was the one interface
        # with no render rate and no latency figure. Off unless asked for; one deque
        # append per frame when on.
        self.metrics = None
        self._metrics_last_emit = 0.0
        if metrics_out or metrics_summary:
            if _MetricsCollector is None:
                print("[PolicyGrid][WARN] perf_metrics not importable; metrics DISABLED.")
            else:
                if metrics_out == "auto":
                    metrics_out = os.path.join(
                        "session_logs",
                        f"metrics_grid_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
                    )
                    os.makedirs("session_logs", exist_ok=True)
                self.metrics = _MetricsCollector(
                    out_path=metrics_out or None,
                    window_s=float(metrics_window_s),
                    print_summary=bool(metrics_summary),
                    session_index=None,
                    mode="desktop_rgb",
                )
                self.metrics.set_label("sessions", int(self.sessions))
                self.metrics.set_label("resolution", f"{self.width}x{self.height}")
                self.metrics.set_label("grid_layout", str(self.grid_layout))
                self.metrics.set_label("observe_only", bool(self.observe_only))
                self.metrics.set_label("vsync", True)
                print(f"[PolicyGrid][Metrics] ON -> {metrics_out or '(summary only)'}")

    def init(self):
        if self.sessions <= 0:
            raise ValueError("--sessions must be > 0")

        # GLFW first: _preferred_offscreen_size queries the monitor, which silently
        # returns nothing (and warns) before glfw.init().
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.model = mujoco.MjModel.from_xml_path(self.xml)
        # Size the offscreen buffer ONCE, to the monitor. The old behaviour grew it
        # reactively on window resize and rebuilt the MjrContext -- which with per-variant
        # slots would mean rebuilding every slot's context (~137 MB of GL uploads each) on
        # a drag. Sizing up front makes _ensure_offscreen_size a no-op in practice.
        self.offscreen_width, self.offscreen_height = self._preferred_offscreen_size()
        self._apply_offscreen_policy(self.model)
        self.viewer = SceneViewer(self.model, view_mode="free")

        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        self.window = glfw.create_window(
            self.width,
            self.height,
            "Policy Grid Viewer",
            None,
            None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_100,
        )
        # The base slot. Pinned: it is what every degraded session renders through, and all
        # overlay text is drawn with its context (mixing font atlases across contexts is a
        # known source of blank text).
        self.slots[BASE_KEY] = VariantSlot(
            key=BASE_KEY, model=self.model, viewer=self.viewer, ctx=self.ctx,
        )
        self._init_variant_cache()

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, PolicyGridViewer._key_callback)
        glfw.set_mouse_button_callback(self.window, PolicyGridViewer._mouse_button_callback)

        import zmq

        self.zmq_ctx = zmq.Context.instance()
        for idx in range(self.sessions):
            sub = self.zmq_ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.LINGER, 0)
            sub.setsockopt(zmq.CONFLATE, 1)
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.connect(f"tcp://{self.state_host}:{self.state_base_port + idx * self.port_step}")

            cmd = self.zmq_ctx.socket(zmq.PUSH)
            cmd.setsockopt(zmq.LINGER, 0)
            cmd.setsockopt(zmq.SNDHWM, 1)
            cmd.connect(f"tcp://{self.cmd_host}:{self.cmd_base_port + idx * self.port_step}")

            self.session_states.append(
                SessionState(
                    index=idx,
                    data=mujoco.MjData(self.model),
                    sub=sub,
                    cmd=cmd,
                )
            )
        self.display_order = list(range(len(self.session_states)))

        self._init_block_logging()

        mode = "observe-only" if self.observe_only else "interactive"
        print(
            f"[PolicyGrid] Started {mode} viewer for {self.sessions} sessions "
            f"state={self.state_host}:{self.state_base_port}+{self.port_step} "
            f"cmd={self.cmd_host}:{self.cmd_base_port}+{self.port_step}"
        )

    def run(self):
        self.init()
        try:
            while not glfw.window_should_close(self.window):
                _m = self.metrics
                _t0 = time.perf_counter()
                glfw.poll_events()
                _t1 = time.perf_counter()
                self._receive_states()
                _t2 = time.perf_counter()
                self._render()
                _t3 = time.perf_counter()
                # swap_buffers BLOCKS on vsync (swap_interval(1)), so this is where the
                # frame budget actually goes. Timing it separately is what distinguishes
                # "the grid is slow" from "the grid is waiting for the display", which the
                # total frame time alone cannot.
                glfw.swap_buffers(self.window)
                _t4 = time.perf_counter()
                if _m is not None:
                    _m.observe("poll_ms", (_t1 - _t0) * 1000.0)
                    _m.observe("state_recv_ms", (_t2 - _t1) * 1000.0)
                    # Deliberately NOT called `render_ms` / `tick_total_ms`. On the VR
                    # side those mean "one camera render" and "one publish tick"; here
                    # they would mean "all N tiles" and "one display frame". Same name,
                    # different quantity, is exactly how a cross-condition table ends up
                    # wrong. Only `state_to_render_ms` below is the same measurement in
                    # both, and that is the one the three interfaces are compared on.
                    _m.observe("grid_render_ms", (_t3 - _t2) * 1000.0)
                    _m.observe("swap_ms", (_t4 - _t3) * 1000.0)
                    _m.observe("frame_ms", (_t4 - _t0) * 1000.0)
                    _m.incr("frames", 1.0)
                    self._observe_state_latency(_t3)
                    _m.maybe_emit()
        finally:
            self.close()

    def close(self):
        # Before the sockets go: the lifetime summary is the headline number for this
        # condition and must survive a Ctrl+C teardown.
        if self.metrics is not None:
            try:
                self.metrics.close()
            except Exception as exc:
                print(f"[PolicyGrid][WARN] metrics close failed: {exc}")
            self.metrics = None
        self.close_block_logging()
        for state in self.session_states:
            for sock in (state.sub, state.cmd):
                try:
                    sock.close(0)
                except Exception:
                    pass
        self.session_states = []

        if self.ctx is not None:
            try:
                self.ctx.free()
            except Exception:
                pass
            self.ctx = None

        if self.window is not None:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None
        glfw.terminate()

    def _observe_state_latency(self, rendered_at: float):
        """Age of the policy state that was just drawn, per the PUBLISHER's clock.

        `state_to_render_ms` is the selected cell only, because that is the one the
        operator is actually looking at and it is the direct analogue of the VR
        single-view number. The whole-grid distribution goes to a separate name so the
        two are never silently averaged together.
        """
        m = self.metrics
        if m is None:
            return
        now = time.time()
        selected = None
        for idx, state in enumerate(self.session_states):
            wall = state.applied_state_wall
            if wall <= 0.0:
                continue
            age_ms = max(0.0, (now - wall) * 1000.0)
            m.observe("state_to_render_ms.grid", age_ms)
            if idx == self.selected_idx:
                selected = age_ms
        if selected is not None:
            m.observe("state_to_render_ms", selected)
        m.set_label("selected_cell", int(self.selected_idx))
        m.set_label("zoomed", bool(self.zoomed))

    def _receive_states(self):
        for state in self.session_states:
            latest = None
            while True:
                try:
                    raw = state.sub.recv(flags=1)
                except Exception:
                    break
                if self.metrics is not None:
                    # Every message off the wire, including the ones superseded before
                    # they are applied -- "arriving at 30 Hz but only 6 Hz of it is
                    # distinct" is a real and reportable distinction.
                    self.metrics.incr("state_msgs", 1.0)
                try:
                    latest = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    print(f"[PolicyGrid][WARN] S{state.index:02d} dropped malformed state: {exc}")

            if latest is None:
                continue

            previous_episode_id = state.episode_id
            next_episode_id = str(latest.get("episode_id", "")) or None
            if self._apply_state(state, latest):
                # FACTR takeover/handback narration. Computed from the PREVIOUS state, so
                # it must run before state.latest is replaced below. Only sessions that
                # publish factr_active=True can trigger it.
                if bool(latest.get("factr_active", False)):
                    previous_live = bool(
                        state.latest is not None
                        and state.latest.get("intervention_live", False)
                    )
                    current_live = bool(latest.get("intervention_live", False))
                    if current_live and not previous_live:
                        print("[FACTR] Target pose reached. You may take over now.")
                    elif previous_live and not current_live:
                        print("[FACTR] Intervention finished. Control returned to policy.")
                state.latest = latest
                state.episode_id = next_episode_id
                state.last_rx_wall = time.time()
                state.stale_warned = False
                # Session identity is spatially stable for the lifetime of the grid.
                # An episode transition must never move a session's panel or make the LIVE
                # indicator appear to jump cells. This matters more now that OOD-ness
                # migrates between sessions: the SCENE changes, the cell never does.

    # ------------------------------------------------------------------ model variants

    # The grid renders ONLY to the window: every mjr_setBuffer in this file selects
    # mjFB_WINDOW, and there is no mjFB_OFFSCREEN or mjr_readPixels anywhere. So the
    # offscreen framebuffer each MjrContext allocates is never rendered into -- at
    # 1920x1080 with the default 4x multisampling that is 66 MB of VRAM per context, 265 MB
    # across four variant slots, on a card the point-cloud workers are already fighting
    # over. Allocate the minimum instead and skip multisampling, which affects the offscreen
    # buffer only and cannot change how the visible window looks.
    _OFFSCREEN_UNUSED_W = 64
    _OFFSCREEN_UNUSED_H = 64

    def _preferred_offscreen_size(self):
        return self._OFFSCREEN_UNUSED_W, self._OFFSCREEN_UNUSED_H

    def _apply_offscreen_policy(self, model):
        model.vis.global_.offwidth = self.offscreen_width
        model.vis.global_.offheight = self.offscreen_height
        try:
            model.vis.quality.offsamples = 0
        except Exception:
            pass

    def _init_variant_cache(self):
        if self.variant_render != "exact":
            return
        try:
            from model_variants.cache import VariantCache
            self.variant_cache = VariantCache(
                self.xml, reference_model=self.model,
                capacity=self.variant_cache_size,
                label="GridVariant", on_evict=self._on_variant_evicted,
            )
            self.variant_cache.install_base(self.model)
        except Exception as exc:
            print(f"[PolicyGrid][WARN] variant cache unavailable ({exc}); "
                  "rendering every session with the base model.")
            self.variant_cache = None
            self.variant_render = "base"

    def _on_variant_evicted(self, key, model):
        """Free the GL objects for an evicted model in the same operation."""
        slot = self.slots.pop(key, None)
        if slot is None:
            return
        try:
            glfw.make_context_current(self.window)
            slot.ctx.free()
        except Exception:
            pass

    def _variant_warn_once(self, key, message):
        if key not in self._variant_warned:
            self._variant_warned.add(key)
            print(message)

    def _ensure_slot(self, key, descriptor):
        """The render slot for a variant, or None if it is not available right now.

        NEVER blocks: `cache.get()` is a dict lookup and `prefetch()` queues a background
        compile. A miss renders degraded with a badge until the worker finishes, which is
        why the wire carries `next_variant` -- with it the grid has a whole episode of
        warning and the badge should essentially never be seen.
        """
        if key == BASE_KEY or self.variant_render != "exact" or self.variant_cache is None:
            return None
        slot = self.slots.get(key)
        if slot is not None:
            slot.last_used = time.time()
            return slot
        model = self.variant_cache.get(key)
        if model is None:
            reason = self.variant_cache.failure_reason(key)
            if reason:
                self._variant_warn_once(
                    f"fail:{key}",
                    f"[PolicyGrid][WARN] variant {key} cannot be built ({reason}); "
                    "rendering that session with the base model.")
                return None
            if descriptor is not None:
                self.variant_cache.prefetch(descriptor)
            return None
        try:
            glfw.make_context_current(self.window)
            self._apply_offscreen_policy(model)
            viewer = SceneViewer(model, view_mode="free")
            ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        except Exception as exc:
            self._variant_warn_once(
                f"slot:{key}",
                f"[PolicyGrid][WARN] could not build a render slot for {key} ({exc}); "
                "rendering that session with the base model.")
            return None
        slot = VariantSlot(key=key, model=model, viewer=viewer, ctx=ctx)
        self.slots[key] = slot
        self.variant_cache.retain(key)
        print(f"[PolicyGrid] variant render slot {key} ready "
              f"({len(self.slots)}/{self.variant_cache_size} slots)")
        return slot

    def _release_unused_slots(self):
        """Drop render slots no session is showing, so the cache stays inside its budget.

        Without this a slot is retained for the lifetime of the process: a session that has
        moved on to a new OOD scene still pins the old model's MjModel (~404 MB RAM) and its
        MjrContext (~137 MB VRAM), and the pinned entries eventually block eviction entirely.
        """
        if self.variant_cache is None:
            return
        # Union of "what the tile last drew" and "what the session now wants": keying on
        # render_key alone frees a slot at the top of the frame that _slot_for recreates a
        # few lines later, once per frame, for a session that never actually moved on.
        live = {st.render_key for st in self.session_states}
        live |= {st.variant_key for st in self.session_states}
        live.add(BASE_KEY)
        for key in [k for k in self.slots if k not in live]:
            slot = self.slots.pop(key, None)
            if slot is None:
                continue
            try:
                glfw.make_context_current(self.window)
                slot.ctx.free()
            except Exception:
                pass
            try:
                self.variant_cache.release(key)
            except Exception:
                pass
            print(f"[PolicyGrid] released variant render slot {key} "
                  f"({len(self.slots)} slot(s) live)")

    def _slot_for(self, state: SessionState):
        """`(slot, degraded)` -- the slot to render this session with.

        Degrading is PER SESSION and never global: one unavailable variant must not change
        how the other eight tiles render, and it must never cause eviction thrashing.
        """
        if state.variant_key == BASE_KEY:
            state.render_key = BASE_KEY
            state.degraded = False
            return self.slots[BASE_KEY], False
        slot = self._ensure_slot(state.variant_key, state.variant_desc)
        if slot is None:
            state.render_key = BASE_KEY
            state.degraded = True
            return self.slots[BASE_KEY], True
        state.render_key = slot.key
        state.degraded = False
        return slot, False

    def _sync_session_variant(self, state: SessionState, message: dict):
        """Adopt the variant a session reports. Returns False only for a REFUSAL.

        The rule that matters: a variant mismatch must never block state application.
        qpos/qvel/ctrl are variant-independent (identical nq/nv/nu, joint order and
        actuator order are enforced when the variant is built), so the state is always safe
        to write -- only the rendering would be wrong. The predecessor of this code ignored
        state it could not interpret and froze the tile forever while still receiving at
        full rate; that must not come back in a new form.
        """
        if int(message.get("version", 1)) < 2:
            state.variant_key = BASE_KEY      # a v1 publisher has no variants
            state.variant_desc = None
            return True

        # A consumer launched on a different scene file would render a plausible-looking
        # lie. That is a launcher bug and has to be loud.
        theirs = str(message.get("base_scene_sha256", "") or "")
        if theirs and self._scene_closure_sha256 and theirs != self._scene_closure_sha256:
            if not state.incompatible:
                state.incompatible = True
                print(f"[PolicyGrid][ERROR] S{state.index:02d} is running a DIFFERENT scene "
                      f"(base_scene_sha256 {theirs[:12]}... != "
                      f"{self._scene_closure_sha256[:12]}...). Refusing its state; the grid "
                      "and that session were launched on different XMLs.")
            return False
        state.incompatible = False

        # DELIBERATELY NOT prefetching `next_variant` here.
        #
        # The hint is worth acting on in the policy and VR processes, which each follow ONE
        # session. The grid follows N. Every cups session publishes a hint (they are all
        # OOD-capable, not just the 2-3 currently OOD), so honouring them meant ~9
        # speculative keys plus up to 3 live ones competing for a 4-entry cache: measured
        # 8 distinct variants and 22 builds, one key recompiled 5 times, ~600 ms each,
        # forever. That starved the state receiver and tiles aged past `stale_timeout_s`
        # and rendered black. The grid has no deadline -- a tile can sit on the base model
        # with a badge for a second -- so it builds only what it is actually rendering.

        key = str(message.get("variant_key", BASE_KEY) or BASE_KEY)
        descriptor = message.get("variant")
        if descriptor is not None:
            ok, reason = is_valid_descriptor(descriptor)
            if not ok:
                self._variant_warn_once(
                    f"desc:{state.index}",
                    f"[PolicyGrid][WARN] S{state.index:02d} sent a malformed variant "
                    f"descriptor ({reason}); rendering with the base model.")
                descriptor, key = None, BASE_KEY
            elif variant_key(descriptor) != key:
                self._variant_warn_once(
                    f"keymismatch:{state.index}",
                    f"[PolicyGrid][WARN] S{state.index:02d} variant_key {key} does not "
                    "match its descriptor; rendering with the base model.")
                descriptor, key = None, BASE_KEY
        state.variant_key = key
        state.variant_desc = descriptor
        state.model_epoch = int(message.get("model_epoch", 0) or 0)
        return True

    def _apply_state(self, state: SessionState, message: dict) -> bool:
        if not self._sync_session_variant(state, message):
            return False
        # Pick the render slot first: an `MjData` belongs to the model it was built from, so
        # when a session's slot changes its data has to be rebuilt too.
        slot, _degraded = self._slot_for(state)
        if state.data_key != slot.key:
            state.data = mujoco.MjData(slot.model)
            state.data_key = slot.key

        qpos = np.asarray(message.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(message.get("qvel", []), dtype=np.float64)
        ctrl = np.asarray(message.get("ctrl", []), dtype=np.float64)
        if (
            qpos.shape != state.data.qpos.shape
            or qvel.shape != state.data.qvel.shape
            or ctrl.shape != state.data.ctrl.shape
        ):
            if not state.stale_warned:
                print(
                    f"[PolicyGrid][WARN] S{state.index:02d} shape mismatch; "
                    f"qpos={qpos.shape}/{state.data.qpos.shape} "
                    f"qvel={qvel.shape}/{state.data.qvel.shape} "
                    f"ctrl={ctrl.shape}/{state.data.ctrl.shape}"
                )
                state.stale_warned = True
            return False

        state.data.qpos[:] = qpos
        state.data.qvel[:] = qvel
        state.data.ctrl[:] = ctrl
        state.data.time = float(message.get("sim_t", state.data.time))
        mujoco.mj_forward(slot.model, state.data)
        state.applied_seq = int(message.get("seq", -1))
        state.applied_state_wall = float(message.get("wall_t", 0.0) or 0.0)
        state.applied_states += 1
        return True

    def _render(self):
        self._release_unused_slots()
        win_w, win_h = glfw.get_framebuffer_size(self.window)
        self._ensure_offscreen_size(win_w, win_h)
        full = mujoco.MjrRect(0, 0, win_w, win_h)

        glfw.make_context_current(self.window)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        mujoco.mjr_rectangle(full, 0.06, 0.06, 0.06, 1.0)

        if self.zoomed:
            self._render_tile(self.selected_idx, full)
            self._draw_global_overlay(full)
            return

        grid_rect, detail_rect = compute_split_layout(win_w, win_h)
        rects = compute_viewports_in_rect(
            grid_rect, len(self.session_states), pad=8, layout=self.grid_layout
        )
        for slot_idx, rect in enumerate(rects):
            self._render_tile(self._session_idx_for_slot(slot_idx), rect)
        self._render_selected_detail(detail_rect)
        self._draw_global_overlay(full)

    def _ensure_offscreen_size(self, win_w: int, win_h: int):
        """No-op: the offscreen buffer is never rendered into (see _preferred_offscreen_size).

        This used to grow the buffer to the window and rebuild the MjrContext. With
        per-variant slots that meant rebuilding EVERY slot's context on a window drag --
        each one re-uploading the scene's meshes and textures -- to resize a buffer nothing
        reads. Window rendering is bounded by the window, not by offwidth/offheight.
        """
        return

    def _render_tile(self, idx: int, rect):
        if idx < 0 or idx >= len(self.session_states):
            return

        state = self.session_states[idx]
        age = time.time() - state.last_rx_wall if state.last_rx_wall > 0 else float("inf")
        stale = age > self.stale_timeout_s

        # Keep rendering the LAST GOOD state while a session is silent. `state.data`
        # retains the last applied qpos/qvel/ctrl indefinitely — nothing clears it on
        # staleness — so there is always something to draw once a first message has
        # arrived. Blanking here is what turned every multi-second stall (robot
        # alignment, a variant compile) into a wall of black tiles; the VR side already
        # keeps publishing the last pose and merely warns. `stale` now only picks the
        # badge, not whether we draw.
        if state.latest is not None:
            try:
                slot, degraded = self._slot_for(state)
                self._render_camera_direct(state.data, rect, slot=slot)
            except Exception as exc:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                mujoco.mjr_rectangle(rect, 0.12, 0.04, 0.04, 1.0)
                self._draw_overlay(rect, f"S{idx:02d} render error", str(exc)[:80])
                return
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
            mujoco.mjr_rectangle(rect, 0.02, 0.02, 0.02, 1.0)

        # All overlays are drawn with the BASE context so text uses one font atlas.
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

        if stale:
            what = "no signal" if state.latest is None else "last frame"
            self._draw_overlay(rect, f"S{idx:02d} {what}", f"age={age:.1f}s")
        else:
            self._draw_session_overlay(idx, rect)

        self._draw_tile_frames(idx, rect, state, stale)
        if idx == self.selected_idx and state.latest is not None:
            self._draw_selected_live_marker(rect)

    def _render_selected_detail(self, rect):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        mujoco.mjr_rectangle(rect, 0.035, 0.035, 0.035, 1.0)
        if not self.session_states:
            self._draw_overlay(rect, "selected RGB", "no sessions")
            return

        state = self.session_states[self.selected_idx]
        age = time.time() - state.last_rx_wall if state.last_rx_wall > 0 else float("inf")
        stale = age > self.stale_timeout_s
        camera_rects = compute_detail_camera_rects(
            rect, pad=10, aspect=self.detail_aspect
        )
        camera_positions = ("front", "left", "right", "wrist")
        for position, cam_name in zip(camera_positions, self.detail_cameras):
            camera_rect = camera_rects[position]
            if state.latest is None:
                mujoco.mjr_rectangle(camera_rect, 0.02, 0.02, 0.02, 1.0)
                self._draw_overlay(camera_rect, cam_name, "no signal")
                continue

            try:
                slot, _degraded = self._slot_for(state)
                self._render_camera_direct(
                    state.data,
                    camera_rect,
                    cam_name=cam_name,
                    roll_side_cameras=self.roll_side_cameras,
                    slot=slot,
                )
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                status = f"age={age:.1f}s" if stale else f"S{self.selected_idx:02d}"
                self._draw_overlay(camera_rect, cam_name, status)
            except Exception as exc:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                mujoco.mjr_rectangle(camera_rect, 0.12, 0.04, 0.04, 1.0)
                self._draw_overlay(camera_rect, cam_name, str(exc)[:80])

    def _draw_session_overlay(self, idx: int, rect):
        state = self.session_states[idx]
        msg = state.latest or {}
        mode = str(msg.get("mode", "?"))
        paused = ""
        if bool(msg.get("paused", False)):
            paused = f" {PAUSED_LABEL}"
        live = " LIVE" if bool(msg.get("intervention_live", False)) else ""
        risk = float(msg.get("acc_risk", 0.0))
        frame_idx = int(msg.get("frame_idx", 0))
        variant = ""
        if state.incompatible:
            variant = " | WRONG SCENE"
        elif state.variant_key != BASE_KEY:
            variant = f" | OOD v{state.variant_key[:8]}"
            if state.degraded and self.variant_render != "off":
                variant += " (base render)"
        # Deliberately NOT adding the success/failure tally here: this line is already
        # clipped at nine tiles (the `frame=`/`risk=` column loses characters), and the
        # tally lives in the verdict band below where there is room for it.
        left = f"S{idx:02d} | {mode}{paused}{live}{variant}"
        right = f"frame={frame_idx}\nrisk={risk:.1f}"
        self._draw_overlay(rect, left, right)
        self._draw_task_verdict(rect, msg)

    # How long a success/failure verdict stays on the tile. An episode ends and the next
    # one starts immediately, so without a hold the banner would flash for one frame and
    # be unreadable -- the whole point is that an operator watching nine tiles can see
    # WHICH cell just failed and WHY without reading nine log files.
    VERDICT_HOLD_S = 6.0

    def _draw_task_verdict(self, rect, msg):
        """Banner across the bottom of a tile naming the last episode outcome + reason."""
        outcome = str(msg.get("task_outcome", "") or "")
        if not outcome:
            return
        age = time.time() - float(msg.get("task_wall_t", 0.0) or 0.0)
        if age < 0 or age > self.VERDICT_HOLD_S:
            return
        if rect.width < 90 or rect.height < 40:
            return

        success = outcome == "success"
        reason = str(msg.get("task_reason", "") or ("done" if success else "?"))
        step = int(msg.get("task_step", 0))
        by = str(msg.get("task_decided_by", "") or "")

        # TWO lines. On a nine-tile grid a single line leaves ~7 characters for the reason
        # after the "FAILURE @217 " prefix, which is useless -- and mjr_overlay hard-clips
        # at the rect with no ellipsis, so the truncation is silent. Giving the reason its
        # own line roughly triples the room. Sized from the real glyph width (measured
        # ~10.4 px/char at mjFONTSCALE_150; 11 leaves a margin).
        tally = f"{int(msg.get('success_count', 0))}/{int(msg.get('failure_count', 0))}"
        head = f"{'SUCCESS' if success else 'FAILURE'} @{step}  S/F {tally}"
        if by and by != "auto":
            head += f"  ({by})"
        room = max(6, int(rect.width - 16) // 11)
        if len(reason) > room:
            # Keep the tail as well as the head: in `cup1_on_cup2_bad_z:-0.158m` the
            # measurement at the end is more informative than the middle of the name.
            keep = room - 1
            left_n = keep // 2
            reason = reason[:left_n] + "~" + reason[len(reason) - (keep - left_n):]
        text = head[:room] + "\n" + reason

        band_h = 56
        band = mujoco.MjrRect(int(rect.left) + 4, int(rect.bottom) + 4,
                              max(1, int(rect.width) - 8), band_h)
        if success:
            mujoco.mjr_rectangle(band, 0.05, 0.45, 0.12, 0.92)
            self._draw_border(band, 0.25, 1.0, 0.4, 1.0, thickness=2)
        else:
            mujoco.mjr_rectangle(band, 0.50, 0.06, 0.06, 0.92)
            self._draw_border(band, 1.0, 0.30, 0.30, 1.0, thickness=2)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            band,
            text,
            "",
            self.ctx,
        )

    def _draw_global_overlay(self, rect):
        mode = "OBSERVE" if self.observe_only else "INTERACTIVE"
        if self.zoomed:
            left = f"{mode} | S{self.selected_idx:02d} zoom"
            right = "Z/Esc: grid"
        else:
            left = f"{mode} | S{self.selected_idx:02d} selected | click selects | Z zoom"
            right = ("Space:pause  Enter:intervene  C:cancel  S/F:mark  R:restart"
                     if not self.observe_only else "commands ignored")
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            rect,
            left,
            right,
            self.ctx,
        )

    def _draw_overlay(self, rect, left: str, right: str):
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect,
            left,
            right,
            self.ctx,
        )

    def _render_camera_direct(
        self,
        data: mujoco.MjData,
        rect,
        cam_name: str | None = None,
        roll_side_cameras: bool = False,
        slot: VariantSlot | None = None,
    ):
        slot = self.slots[BASE_KEY] if slot is None else slot
        cam_name = self.cam if cam_name is None else cam_name
        # Camera state stays GLOBAL: a per-slot camera would make zoom/orbit behave
        # differently depending on which session you happen to be looking at.
        slot.viewer.cam = self.viewer.cam
        slot.viewer.cam_default = self.viewer.cam_default
        slot.viewer.opt = self.viewer.opt
        slot.viewer.view_mode = self.viewer.view_mode
        slot.viewer._set_fixed_camera(cam_name)
        mujoco.mjv_updateScene(
            slot.model,
            data,
            slot.viewer.opt,
            None,
            slot.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            slot.viewer.scn,
        )

        if self._should_roll_camera_180(cam_name, roll_side_cameras=roll_side_cameras):
            self._roll_scene_camera_180(slot.viewer.scn)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, slot.ctx)
        mujoco.mjr_render(rect, slot.viewer.scn, slot.ctx)

    def _should_roll_camera_180(self, cam_name: str, *, roll_side_cameras: bool = False) -> bool:
        if self.flip_front and cam_name == "front":
            return True
        return roll_side_cameras and cam_name in DETAIL_SIDE_ROLL_180_CAMERAS

    def _roll_scene_camera_180(self, scn=None):
        # Display-only correction for cameras AUTHORED rolled in the XML (up = -Z).
        # In the current scenes that is `front` alone; `left`/`right` are authored
        # upright, which is why rolling them here made the detail panes upside down.
        # The compiled fixed camera has already been converted into MjvScene cameras by
        # mjv_updateScene, so adjust the render camera itself: negating "up" rolls the
        # image 180 degrees while preserving view direction.
        scn = self.viewer.scn if scn is None else scn
        for gl_camera in scn.camera:
            gl_camera.up[:] *= -1.0

    def _draw_tile_frames(self, idx: int, rect, state: SessionState, stale: bool):
        # Borders now survive staleness. The red intervention border in particular is
        # exactly what the operator needs WHILE a takeover is starting up — which is the
        # one moment the session is most likely to be behind on publishing.
        if state.latest is not None:
            self._draw_border(rect, *risk_color(state.latest.get("acc_risk", 0.0)), thickness=6)

        if idx == self.selected_idx:
            self._draw_border(rect, 0.1, 0.8, 1.0, 1.0, thickness=3, inset=7)

        if state.latest is not None and bool(state.latest.get("intervention_live", False)):
            self._draw_border(rect, 1.0, 0.1, 0.1, 1.0, thickness=4, inset=11)

    def _draw_selected_live_marker(self, rect):
        if rect.width < 70 or rect.height < 34:
            return

        label_w = min(86, max(66, int(rect.width) - 20))
        label_h = 28
        left = int(rect.left) + max(8, (int(rect.width) - label_w) // 2)
        bottom = int(rect.bottom) + 12
        label_rect = mujoco.MjrRect(left, bottom, label_w, label_h)
        dot_size = 7
        dot_left = left + 10
        dot_bottom = bottom + (label_h - dot_size) // 2

        mujoco.mjr_rectangle(label_rect, 0.56, 0.0, 0.0, 0.92)
        self._draw_border(label_rect, 1.0, 0.18, 0.18, 1.0, thickness=1)
        mujoco.mjr_rectangle(
            mujoco.MjrRect(dot_left - 1, dot_bottom, dot_size + 2, dot_size),
            1.0,
            0.16,
            0.16,
            1.0,
        )
        mujoco.mjr_rectangle(
            mujoco.MjrRect(dot_left, dot_bottom - 1, dot_size, dot_size + 2),
            1.0,
            0.16,
            0.16,
            1.0,
        )
        self._draw_badge_text(label_rect, "LIVE")

    def _draw_badge_text(self, rect, text: str):
        pixel_font = {
            "L": ("100", "100", "100", "100", "111"),
            "I": ("111", "010", "010", "010", "111"),
            "V": ("10001", "10001", "01010", "01010", "00100"),
            "E": ("111", "100", "110", "100", "111"),
        }
        scale = 3
        gap = 1
        x = int(rect.left) + 34
        y = int(rect.bottom) + 5
        for char in text:
            glyph = pixel_font.get(char.upper())
            if glyph is None:
                x += 2 * scale
                continue
            glyph_w = len(glyph[0])
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit != "1":
                        continue
                    cell_x = x + col * scale
                    cell_y = y + (len(glyph) - 1 - row) * scale
                    mujoco.mjr_rectangle(
                        mujoco.MjrRect(cell_x, cell_y, scale, scale),
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    )
            x += (glyph_w + gap) * scale

    def _draw_border(self, rect, r: float, g: float, b: float, a: float, *, thickness: int, inset: int = 0):
        t = max(1, int(thickness))
        pad = max(0, int(inset))
        left = int(rect.left) + pad
        bottom = int(rect.bottom) + pad
        width = int(rect.width) - 2 * pad
        height = int(rect.height) - 2 * pad
        color = (r, g, b, a)
        if width <= 0 or height <= 0:
            return
        t = min(t, width, height)
        mujoco.mjr_rectangle(mujoco.MjrRect(left, bottom, width, t), *color)
        mujoco.mjr_rectangle(mujoco.MjrRect(left, bottom + height - t, width, t), *color)
        mujoco.mjr_rectangle(mujoco.MjrRect(left, bottom, t, height), *color)
        mujoco.mjr_rectangle(mujoco.MjrRect(left + width - t, bottom, t, height), *color)

    # ------------------------------------------------------------ block recording

    def _init_block_logging(self):
        """Attach to the running experiment block, if there is one.

        Failure is never fatal: the grid is an operator tool first. A block that loses
        the desktop CELL_SELECTED events is degraded, not corrupt -- the policy-side
        events still record everything else.
        """
        try:
            from data_io.experiment_block import (
                BlockEventLog, block_origin, resolve_block_from_env,
            )

            block = resolve_block_from_env()
            if block is None:
                return
            mono_origin, wall_origin = block_origin(block.block_dir)
            self._block = block
            self._block_log = BlockEventLog(
                block.events_path, block=block, source="desktop_grid",
                mono_origin=mono_origin, wall_origin=wall_origin,
            )
            print(f"[PolicyGrid] Block recording ACTIVE -> {block.events_path}")
        except Exception as exc:
            print(f"[PolicyGrid][WARN] block event logging unavailable: {exc}")

    def _log_cell_selected(self, cell_id: int, *, method: str = "mouse"):
        """One CELL_SELECTED per CONFIRMED selection change.

        Only the confirmation is recorded -- never the continuous cursor position.
        The analysis derives inspection time from the gap between this and the
        following INTERVENTION_REQUEST, which needs the confirmation instant only.
        """
        if self._block_log is None:
            return
        try:
            self._block_log.write(
                "CELL_SELECTED", cell_id=int(cell_id),
                selected_cell_id=int(cell_id), selection_method=str(method),
            )
        except Exception as exc:
            print(f"[PolicyGrid][WARN] CELL_SELECTED write failed: {exc}")

    def _notify_selection_to_cells(self, new_idx: int, old_idx):
        """Tell the policy processes which cell is selected (bookkeeping only).

        app.py stores it as a per-frame column; no control path reads it. Suppressed in
        observe-only mode, where the Quest owns selection and the grid must stay silent.
        """
        if self.observe_only or not self.session_states:
            return
        try:
            if old_idx is not None and old_idx != new_idx and 0 <= old_idx < len(self.session_states):
                self._send_command_to_state(self.session_states[old_idx], "DESELECTED",
                                            reason="selection moved")
            self._send_command_to_state(self.session_states[new_idx], "SELECTED",
                                        reason="mouse select")
        except Exception as exc:
            print(f"[PolicyGrid][WARN] selection notify failed: {exc}")

    def _cancel_intervention_on(self, state, *, reason: str) -> bool:
        """Cancel a takeover that is live on a cell the operator is walking away from.

        An intervention holds the real arm in HUMAN_CONTROL and freezes that cell's
        episode, so leaving it live would strand the arm. `CANCEL` is the same command
        the grip button sends, reusing the existing cancel path.

        THE GUARDS BELOW ARE NOT OPTIONAL. `app.py::_handle_policy_command` gives CANCEL
        a second meaning: while a return-home worker is running it ABORTS THE HOMING and
        leaves the arm wherever it stopped. The mirrored state arrives over a CONFLATE
        socket, so `intervention_live` stays True for a moment after a takeover has
        actually finished -- and finishing is exactly when return-home starts. Acting on
        that stale flag meant: finish an intervention, click another cell, and the arm
        halts mid-homing. So we cancel ONLY on state that is fresh, still in replan mode,
        and not already on its way home.
        """
        latest = getattr(state, "latest", None) or {}
        if not bool(latest.get("intervention_live", False)):
            return False

        age = time.time() - state.last_rx_wall if state.last_rx_wall > 0 else float("inf")
        if age > self.stale_timeout_s:
            print(f"[PolicyGrid] S{state.index:02d} looks live but its state is "
                  f"{age:.1f}s stale; not cancelling on a guess.")
            return False
        if str(latest.get("mode", "")) != "replan":
            return False
        if str(latest.get("intervention_phase", "")) == "returning_home":
            # Already ending under its own power. Cancelling here would abort the homing.
            return False

        self._send_command_to_state(state, "CANCEL", reason=reason)
        print(f"[PolicyGrid] S{state.index:02d} had a LIVE intervention; cancelled it "
              f"because the operator selected another cell.")
        return True

    def select_cell(self, idx: int, *, method: str = "mouse"):
        """The single place a confirmed selection is applied AND recorded."""
        previous = self.selected_idx if self._selection_logged else None
        changed = (not self._selection_logged) or (idx != self.selected_idx)
        self.selected_idx = idx
        if not changed:
            return
        self._selection_logged = True
        self._log_cell_selected(idx, method=method)
        # Before the selection notification, so the cancelled cell is still identified
        # as the one being left. Suppressed in observe-only mode, where the Quest owns
        # both the selection and the intervention and the grid must not send commands.
        if (not self.observe_only and previous is not None and previous != idx
                and 0 <= previous < len(self.session_states)):
            self._cancel_intervention_on(self.session_states[previous],
                                         reason="operator selected another cell")
        self._notify_selection_to_cells(idx, previous)

    def close_block_logging(self):
        if self._block_log is not None:
            self._block_log.close()
            self._block_log = None

    def _pick_tile(self, x: float, y: float):
        if self.zoomed:
            return self.selected_idx

        win_w, win_h = glfw.get_framebuffer_size(self.window)
        grid_rect, _ = compute_split_layout(win_w, win_h)
        rects = compute_viewports_in_rect(
            grid_rect, len(self.session_states), pad=8, layout=self.grid_layout
        )
        y_mj = win_h - y
        for idx, rect in enumerate(rects):
            if (
                rect.left <= x <= rect.left + rect.width
                and rect.bottom <= y_mj <= rect.bottom + rect.height
            ):
                return self._session_idx_for_slot(idx)
        return None

    def _session_idx_for_slot(self, slot_idx: int) -> int:
        if len(self.display_order) != len(self.session_states):
            self.display_order = list(range(len(self.session_states)))
        if 0 <= slot_idx < len(self.display_order):
            return self.display_order[slot_idx]
        return slot_idx

    def _toggle_zoom(self):
        if self.session_states:
            self.zoomed = not self.zoomed

    def _send_command(self, cmd: str):
        if not self.session_states:
            return
        if self.observe_only:
            if not self._observe_hint_printed:
                print("[PolicyGrid] Observe-only mode: command keys are ignored.")
                self._observe_hint_printed = True
            return

        state = self.session_states[self.selected_idx]
        self._send_command_to_state(state, cmd)

    def _send_command_to_state(self, state: SessionState, cmd: str, *, reason: str = ""):
        try:
            import zmq

            state.cmd.send_string(cmd, flags=zmq.NOBLOCK)
            suffix = f" ({reason})" if reason else ""
            print(f"[PolicyGrid] S{state.index:02d} -> {cmd}{suffix}")
        except Exception as exc:
            print(f"[PolicyGrid][WARN] failed to send {cmd} to S{state.index:02d}: {exc}")

    @staticmethod
    def _key_callback(window, key, scancode, action, mods):
        app = glfw.get_window_user_pointer(window)
        if app is None or action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if app.zoomed:
                app.zoomed = False
            else:
                glfw.set_window_should_close(window, True)
            return

        if key == glfw.KEY_Z and action == glfw.PRESS:
            app._toggle_zoom()
            return

        if action != glfw.PRESS:
            return

        # Desktop-RGB is the one condition where this keyboard drives the study, so the
        # two bindings that matter are the two an operator uses constantly:
        #   SPACE = pause / resume        ENTER = start / finish an intervention
        # P and X are kept as aliases (X mirrors the Quest's LEFT X button, P is what the
        # muscle memory of earlier runs expects); Enter is the documented key.
        #
        # SEMANTIC NAMES ONLY. Space used to send the bare letter "A", which resolves to
        # sim_toggle purely by an alias three layers away in app.py. That is the exact
        # shape of the bug where Unity's "B" meant sim_toggle to runtime_impl and
        # mark_task_success to app.py, so pressing pause saved the episode as a success
        # and started a new scene. A letter carries no meaning of its own; a name does.
        command_by_key = {
            glfw.KEY_SPACE: "SIM_TOGGLE",
            glfw.KEY_ENTER: "INTERVENE",
            glfw.KEY_KP_ENTER: "INTERVENE",
            glfw.KEY_P: "INTERVENE",
            glfw.KEY_X: "INTERVENE",
            glfw.KEY_C: "CANCEL",
            glfw.KEY_S: "TASK_SUCCESS",
            glfw.KEY_F: "TASK_FAIL",
            glfw.KEY_R: "RESTART",
        }
        cmd = command_by_key.get(key)
        if cmd is not None:
            app._send_command(cmd)

    @staticmethod
    def _mouse_button_callback(window, button, action, mods):
        app = glfw.get_window_user_pointer(window)
        if app is None or button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return

        x, y = glfw.get_cursor_pos(window)
        idx = app._pick_tile(x, y)
        if idx is None:
            return

        now = time.time()
        is_double = app.last_click_idx == idx and (now - app.last_click_wall) <= 0.35
        app.select_cell(idx, method="mouse")
        app.last_click_idx = idx
        app.last_click_wall = now
        if is_double:
            app._toggle_zoom()


def parse_args():
    parser = argparse.ArgumentParser(description="Mirror N policy processes into one front-camera grid UI.")
    parser.add_argument("--xml", required=True, help="MuJoCo XML path used by every policy process.")
    parser.add_argument("--sessions", type=int, required=True, help="Number of policy sessions to mirror.")
    parser.add_argument("--state_host", default="127.0.0.1")
    parser.add_argument("--state_base_port", type=int, default=8066)
    parser.add_argument("--cmd_host", default="127.0.0.1")
    parser.add_argument("--cmd_base_port", type=int, default=8065)
    parser.add_argument("--port_step", type=int, default=10)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--cam", default="front")
    parser.add_argument(
        "--detail_cameras",
        default=",".join(DETAIL_CAMERAS),
        help="Comma-separated detail cameras in order: top-left,top-right,bottom-left,bottom-right.",
    )
    parser.add_argument(
        "--detail_aspect",
        type=float,
        default=DETAIL_ASPECT,
        help="Width/height the detail panes are letterboxed to so the FULL camera image "
             "is visible (mjr_render crops horizontally to the viewport aspect). "
             "0 disables the fit and fills the cell.",
    )
    parser.add_argument("--flip_front", dest="flip_front", action="store_true", default=True)
    parser.add_argument("--no_flip_front", dest="flip_front", action="store_false")
    # OFF by default: the scene XMLs define left/right with up=+Z (upright), so rolling
    # them here displayed the detail panes upside down. Only `front` is authored rolled
    # (up=-Z, for the VR texture path) and it keeps its own --flip_front correction.
    # Turn this on for a scene whose side cameras ARE authored rolled.
    parser.add_argument("--roll_side_cameras", dest="roll_side_cameras",
                        action="store_true", default=False)
    parser.add_argument("--no_roll_side_cameras", dest="roll_side_cameras",
                        action="store_false")
    parser.add_argument("--observe_only", action="store_true")
    parser.add_argument("--stale_timeout_s", type=float, default=1.0)
    parser.add_argument(
        "--metrics_out", default=os.environ.get("GRID_METRICS_OUT", ""),
        help="JSONL sink for Desktop-RGB measurements. 'auto' picks "
             "session_logs/metrics_grid_<ts>.jsonl. Same row schema as the VR runtime's "
             "metrics, so tools/perf_report.py reads it unchanged.",
    )
    parser.add_argument("--metrics_window_s", type=float,
                        default=float(os.environ.get("GRID_METRICS_WINDOW_S", "10")))
    parser.add_argument("--metrics_summary",
                        action="store_true",
                        default=parse_bool(os.environ.get("GRID_METRICS_SUMMARY", "0")))
    parser.add_argument(
        "--grid_layout", choices=(GRID_LAYOUT_VR, GRID_LAYOUT_COMPACT),
        default=os.environ.get("GRID_UI_LAYOUT", GRID_LAYOUT_VR),
        help="cell arrangement. vr (default) = 3 rows filled column-major, identical to "
             "the VR selector grid, so a session sits in the same cell in both interfaces; "
             "compact = the legacy near-square sqrt layout filled row-major.",
    )
    parser.add_argument(
        "--variant_render", choices=("exact", "base", "off"),
        default=os.environ.get("GRID_UI_VARIANT_RENDER", "exact"),
        help="how to render sessions running a model-level OOD variant. exact = compile a "
             "render slot per variant (~404 MB RAM + ~137 MB VRAM each); base = render them "
             "with the base model plus a badge (free); off = base, unbadged.",
    )
    parser.add_argument(
        "--variant_cache", type=int,
        default=int(os.environ.get("OOD_VARIANT_CACHE", "4") or 4),
        help="max compiled models held for rendering, INCLUDING the pinned base slot.",
    )
    # NOTE: the OOD auto-pause supervisor (--ood_enabled / --acc_ood_threshold /
    # --max_active_ood_cells / --ood_evaluate_interval_s / --ood_owner /
    # --ood_manual_override_s) was removed. OOD now means "this session runs an
    # out-of-distribution SCENE", decided per session in app.py — it never pauses anything,
    # so this viewer has no OOD role at all. See OOD_MIN_STATES / OOD_MAX_STATES.
    return parser.parse_args()


def main():
    args = parse_args()
    detail_cameras = tuple(cam.strip() for cam in args.detail_cameras.split(",") if cam.strip())
    if len(detail_cameras) != 4:
        raise SystemExit("--detail_cameras must contain exactly 4 comma-separated camera names")
    app = PolicyGridViewer(
        xml=args.xml,
        sessions=args.sessions,
        state_host=args.state_host,
        state_base_port=args.state_base_port,
        cmd_host=args.cmd_host,
        cmd_base_port=args.cmd_base_port,
        port_step=args.port_step,
        width=args.width,
        height=args.height,
        cam=args.cam,
        detail_cameras=detail_cameras,
        detail_aspect=args.detail_aspect,
        flip_front=args.flip_front,
        roll_side_cameras=args.roll_side_cameras,
        observe_only=args.observe_only,
        stale_timeout_s=args.stale_timeout_s,
        metrics_out=args.metrics_out,
        metrics_window_s=args.metrics_window_s,
        metrics_summary=args.metrics_summary,
        variant_render=args.variant_render,
        variant_cache_size=args.variant_cache,
        grid_layout=args.grid_layout,
    )
    app.run()


if __name__ == "__main__":
    main()
