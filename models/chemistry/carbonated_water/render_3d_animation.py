#!/usr/bin/env pvpython
# -*- coding: utf-8 -*-
"""
ParaView rendering script (5.9.x).

Loads ``state.pvsm`` while remapping the embedded Windows data path to the
local Linux ``vtk_files`` directory, equalizes the camera across all render
views (the first view's camera is used as the master), then renders a video
where the camera slowly rotates around the data z-axis while the animation
steps through each snapshot in the state file.

Run with ParaView's Python interpreter (offscreen by default with pvbatch):
    pvbatch render_3d_animation.py
or interactively:
    pvpython render_3d_animation.py
"""

from __future__ import print_function

import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def _pdeathsig_preexec():
    """
    Linux preexec_fn: send SIGTERM to the child when its parent dies.

    Lets the spawned X server reliably go away when pvbatch exits, no matter
    how chaotic ParaView's interpreter shutdown is.
    """
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        SIGTERM = 15
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0)
    except Exception:
        pass


def _ensure_display():
    """
    Ensure a usable X DISPLAY exists; spawn Xvnc/Xvfb headlessly if not.

    Required for ParaView 5.9 X11 builds that can't render offscreen on their
    own. Picks the first free display number in [99, 199]. The child is set
    up with PR_SET_PDEATHSIG so it dies with us — no atexit / explicit teardown
    needed (and no race with ParaView's own X teardown that produced an XIO
    error 22 + heap-corruption SIGABRT in earlier runs).
    """
    if os.environ.get("DISPLAY"):
        return None
    for cand in (shutil.which("Xvfb"), shutil.which("Xvnc")):
        if cand:
            xbin = cand
            break
    else:
        return None
    for n in range(99, 200):
        lock = "/tmp/.X{}-lock".format(n)
        if os.path.exists(lock):
            continue
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.bind("\0/tmp/.X11-unix/X{}-probe-{}".format(n, os.getpid()))
        except OSError:
            pass
        if "Xvfb" in os.path.basename(xbin):
            cmd = [xbin, ":{}".format(n), "-screen", "0", "1920x1080x24",
                   "-nolisten", "tcp"]
        else:
            cmd = [xbin, ":{}".format(n), "-geometry", "1920x1080",
                   "-depth", "24", "-SecurityTypes", "None", "-NeverShared",
                   "-AlwaysShared=0", "-localhost"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=_pdeathsig_preexec,
            )
        except Exception:
            continue
        time.sleep(1.0)
        if proc.poll() is not None:
            continue
        os.environ["DISPLAY"] = ":{}".format(n)
        return proc
    return None


_xserver = _ensure_display()
if _xserver is not None:
    print("[render_animation] Started virtual X server on DISPLAY={}".format(
        os.environ.get("DISPLAY")))
    sys.stdout.flush()


from paraview.simple import (
    GetAnimationScene,
    GetLayout,
    GetRenderViews,
    GetSources,
    GetTimeKeeper,
    LoadState,
    Render,
    SaveScreenshot,
)


# ============================================================================
# Configuration
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

STATE_FILE = os.path.join(HERE, "state.pvsm")

# Directory containing solution.pvd / solution_ts*.vtu / mesh.vtk
DATA_DIR = os.path.join(
    HERE,
    "output_3D_195k_cpu_calcite_3_0.1_ts_0.002_phreeqc_phreeqc_2",
    "vtk_files",
)

OUTPUT_DIR = os.path.join(HERE, "render_output")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
# Final video path; encoder is chosen from the extension.
# Supported here: ".ogv" (vtkOggTheoraWriter, always available with ParaView)
# or ".mp4" (requires ffmpeg on PATH or via FFMPEG override).
VIDEO_FILE = os.path.join(OUTPUT_DIR, "animation.mp4")

# Optional explicit ffmpeg binary; otherwise we look it up on PATH.
# This box has no system ffmpeg, so we point at the binary shipped with
# the imageio-ffmpeg wheel installed in the 'chemistry' conda env.
FFMPEG = "/oahu/data/avnovikov/mambaforge/envs/chemistry/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

# Target total video length, in seconds. When set, FRAMES_PER_SNAPSHOT below
# is ignored and recomputed from VIDEO_LENGTH_SEC * FRAME_RATE / n_timesteps
# (clamped to >= 1 frame/snapshot). Set to None to use FRAMES_PER_SNAPSHOT as
# a literal value.
VIDEO_LENGTH_SEC = 5

# How many rotated frames to render per snapshot (time step). Used only when
# VIDEO_LENGTH_SEC is None.
FRAMES_PER_SNAPSHOT = 5

# Degrees of rotation around the z-axis per snapshot.
DEG_PER_SNAPSHOT = 15.0

# Output resolution for each screenshot.
IMAGE_RESOLUTION = [1920, 1080]

# Playback frame rate of the final video.
FRAME_RATE = 30

# Wipe any existing PNGs in FRAMES_DIR before rendering.
CLEAN_FRAMES_DIR = True

# Which RenderView to use as the camera master (0 = first).
MASTER_VIEW_INDEX = 0

# None -> rotation axis is a z-line through the master's CameraFocalPoint.
# Set to [x, y, z] to override (only x, y are used).
ROTATION_CENTER = None

# ----------------------------------------------------------------------------
# Threshold filter overrides.
#
# After the .pvsm is loaded we walk every Threshold filter and, if its scalar
# input matches a key here, we override its (lower, upper) bounds. Defaults
# below mirror the values currently saved in state.pvsm — leave them alone to
# reproduce the existing visualization, edit a row to widen or narrow a band.
# Set the value to None to skip an override (use the value from state.pvsm).
# ----------------------------------------------------------------------------
THRESHOLD_BOUNDS = {
    "porosity":  (0.55, 1.0),   # Threshold #1 — porosity field
    "SR_CaCO3":  (0.0, 0.9),   # Threshold #2 — saturation ratio of calcite
    "x_CH4":     (0.1, 0.1),   # Threshold #3 — methane mole fraction
}


# ============================================================================
# Helpers
# ============================================================================


def log(msg):
    print("[render_animation] {}".format(msg))
    sys.stdout.flush()


def rewrite_state_paths(state_path, data_dir):
    """
    Rewrite absolute file paths inside a .pvsm so they point at ``data_dir``.

    The Windows/Linux state files store FileName values as absolute paths.
    This helper replaces the path of any reference whose basename exists in
    ``data_dir`` with the local absolute path.

    Returns the path to a temporary, rewritten copy of the state file.
    """
    with open(state_path, "r") as f:
        text = f.read()

    available = set(os.listdir(data_dir))
    pattern = re.compile(r'value="([^"]*[\\/]([^"\\/]+))"')
    replacements = {}

    def _maybe_replace(match):
        full = match.group(1)
        base = match.group(2)
        if base in available:
            new = os.path.join(data_dir, base)
            replacements.setdefault(full, new)
            return 'value="{}"'.format(new)
        return match.group(0)

    new_text = pattern.sub(_maybe_replace, text)

    if not replacements:
        log("No path rewrites needed inside {}.".format(state_path))
        return state_path

    for old, new in replacements.items():
        log("  rewrite: {} -> {}".format(old, new))

    fd, tmp = tempfile.mkstemp(prefix="state_remapped_", suffix=".pvsm")
    os.close(fd)
    with open(tmp, "w") as f:
        f.write(new_text)
    log("Wrote rewritten state to {}".format(tmp))
    return tmp


def rotate_point_around_z(px, py, pz, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    nx = cx + dx * c - dy * s
    ny = cy + dx * s + dy * c
    return [nx, ny, pz]


def rotate_vec_around_z(vx, vy, vz, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [vx * c - vy * s, vx * s + vy * c, vz]


def copy_camera(src, dst):
    dst.CameraPosition = list(src.CameraPosition)
    dst.CameraFocalPoint = list(src.CameraFocalPoint)
    dst.CameraViewUp = list(src.CameraViewUp)
    dst.CameraViewAngle = src.CameraViewAngle
    dst.CameraParallelScale = src.CameraParallelScale
    dst.CameraParallelProjection = src.CameraParallelProjection
    try:
        dst.CenterOfRotation = list(src.CenterOfRotation)
    except Exception:
        pass


def set_camera_on_views(views, position, focal, viewup):
    for v in views:
        v.CameraPosition = list(position)
        v.CameraFocalPoint = list(focal)
        v.CameraViewUp = list(viewup)


def apply_threshold_overrides():
    """
    Override (lower, upper) bounds of any Threshold filter whose scalar
    matches a key in THRESHOLD_BOUNDS.
    """
    if not THRESHOLD_BOUNDS:
        return
    found = 0
    overridden = 0
    for name, proxy in GetSources().items():
        try:
            xml_name = proxy.SMProxy.GetXMLName()
        except Exception:
            continue
        if xml_name != "Threshold":
            continue
        found += 1
        # The "SelectInputScalars" property is a 5-tuple in the saved state
        # (..., association, array_name). The Python wrapper's attribute view
        # often collapses this to None or a 2-tuple, so read it via the SM API
        # directly to make the array name available.
        field = None
        try:
            sm = proxy.SMProxy.GetProperty("SelectInputScalars")
            n = sm.GetNumberOfElements() if sm is not None else 0
            if n:
                field = sm.GetElement(n - 1) or None
        except Exception:
            field = None
        if not field:
            try:
                scalars = list(proxy.SelectInputScalars)
                field = scalars[-1] if scalars else None
            except Exception:
                field = None
        if not field or field not in THRESHOLD_BOUNDS:
            log("  threshold filter '{}' on '{}' has no override".format(name, field))
            continue
        bounds = THRESHOLD_BOUNDS[field]
        if bounds is None:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        sm_prop = proxy.SMProxy.GetProperty("ThresholdBetween")
        if sm_prop is None:
            log("  threshold[{}]: no ThresholdBetween property; skipping".format(field))
            continue
        current = [sm_prop.GetElement(0), sm_prop.GetElement(1)]
        sm_prop.SetElement(0, lo)
        sm_prop.SetElement(1, hi)
        try:
            proxy.SMProxy.UpdateVTKObjects()
        except Exception:
            pass
        try:
            proxy.UpdatePipeline()
        except Exception:
            pass
        overridden += 1
        log("  threshold[{}]: {} -> [{}, {}]".format(field, current, lo, hi))
    log("Threshold filters: {} found, {} overridden".format(found, overridden))


def collect_timesteps():
    tk = GetTimeKeeper()
    steps = []
    try:
        tv = tk.TimestepValues
        if tv is None:
            steps = []
        else:
            try:
                steps = list(tv)
            except TypeError:
                steps = [float(tv)]
    except Exception:
        steps = []
    if not steps:
        log("No time steps in TimeKeeper; rendering a single frame.")
        steps = [None]
    return steps


def set_animation_time(t):
    if t is None:
        return
    scene = GetAnimationScene()
    scene.AnimationTime = t
    try:
        GetTimeKeeper().Time = t
    except Exception:
        pass


def clean_frames():
    if not CLEAN_FRAMES_DIR:
        return
    if not os.path.isdir(FRAMES_DIR):
        return
    for f in os.listdir(FRAMES_DIR):
        if f.lower().endswith(".png"):
            try:
                os.remove(os.path.join(FRAMES_DIR, f))
            except OSError:
                pass


def find_ffmpeg():
    if FFMPEG and os.path.isfile(FFMPEG):
        return FFMPEG
    return shutil.which("ffmpeg")


def encode_with_ffmpeg(out_path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    frames_glob = os.path.join(FRAMES_DIR, "frame_%05d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate", str(FRAME_RATE),
        "-i", frames_glob,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        out_path,
    ]
    log("Encoding video with ffmpeg: {}".format(" ".join(cmd)))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
        rc = proc.returncode
    except Exception as e:
        log("ffmpeg invocation failed: {}".format(e))
        return False
    if rc != 0:
        try:
            log("ffmpeg failed (rc={}):\n{}".format(rc, err.decode("utf-8", "replace")))
        except Exception:
            log("ffmpeg failed (rc={}).".format(rc))
        return False
    log("Video written to: {}".format(out_path))
    return True


def encode_with_vtk_ogg(out_path):
    """
    Encode rendered PNGs to .ogv via vtkOggTheoraWriter (no ffmpeg needed).
    """
    try:
        from vtkmodules.vtkIOImage import vtkPNGReader
        from vtkmodules.vtkIOOggTheora import vtkOggTheoraWriter
    except Exception as e:
        log("vtkOggTheoraWriter not available: {}".format(e))
        return False

    pngs = sorted(
        os.path.join(FRAMES_DIR, f)
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith(".png")
    )
    if not pngs:
        log("No PNG frames to encode.")
        return False

    reader = vtkPNGReader()
    reader.SetFileName(pngs[0])
    reader.Update()

    writer = vtkOggTheoraWriter()
    writer.SetFileName(out_path)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetRate(int(FRAME_RATE))
    try:
        writer.SetQuality(2)
    except Exception:
        pass
    writer.Start()
    for i, p in enumerate(pngs):
        reader.SetFileName(p)
        reader.Update()
        writer.Write()
        if (i + 1) % 30 == 0:
            log("  encoded {}/{} frames".format(i + 1, len(pngs)))
    writer.End()
    log("Video written to: {}".format(out_path))
    return True


def encode_video():
    ext = os.path.splitext(VIDEO_FILE)[1].lower()
    if ext == ".mp4":
        if encode_with_ffmpeg(VIDEO_FILE):
            return True
        log("ffmpeg unavailable; falling back to .ogv via vtkOggTheoraWriter.")
        ogv_path = os.path.splitext(VIDEO_FILE)[0] + ".ogv"
        return encode_with_vtk_ogg(ogv_path)
    if ext == ".ogv":
        if encode_with_vtk_ogg(VIDEO_FILE):
            return True
        log("vtkOggTheoraWriter failed; trying ffmpeg .mp4 fallback.")
        mp4_path = os.path.splitext(VIDEO_FILE)[0] + ".mp4"
        return encode_with_ffmpeg(mp4_path)
    log("Unsupported video extension '{}'; only .ogv and .mp4 are wired up.".format(ext))
    return False


# ============================================================================
# Main
# ============================================================================


def main():
    if not os.path.isfile(STATE_FILE):
        log("ERROR: state file not found: {}".format(STATE_FILE))
        sys.exit(1)
    if not os.path.isdir(DATA_DIR):
        log("ERROR: data dir not found: {}".format(DATA_DIR))
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    clean_frames()

    log("Remapping state file paths to: {}".format(DATA_DIR))
    remapped_state = rewrite_state_paths(STATE_FILE, DATA_DIR)

    log("Loading state: {}".format(remapped_state))
    LoadState(remapped_state)

    # Belt-and-braces: fix any source FileName that still points at a stale path
    # (covers proxies whose value attribute we missed, e.g. file lists).
    available = set(os.listdir(DATA_DIR))
    for proxy in GetSources().values():
        fn_prop = proxy.GetProperty("FileName") if hasattr(proxy, "GetProperty") else None
        if fn_prop is None:
            continue
        try:
            current = proxy.FileName
        except Exception:
            continue
        if isinstance(current, str):
            base = os.path.basename(current.replace("\\", "/"))
            if base in available:
                new = os.path.join(DATA_DIR, base)
                if new != current:
                    log("  source FileName: {} -> {}".format(current, new))
                    proxy.FileName = new
                    try:
                        proxy.UpdatePipeline()
                    except Exception:
                        pass

    apply_threshold_overrides()

    render_views = list(GetRenderViews())
    if not render_views:
        log("ERROR: no render views found in state.")
        sys.exit(2)

    log("Found {} render view(s).".format(len(render_views)))
    master_idx = MASTER_VIEW_INDEX
    if master_idx < 0 or master_idx >= len(render_views):
        log("MASTER_VIEW_INDEX {} out of range; falling back to 0.".format(master_idx))
        master_idx = 0
    master = render_views[master_idx]
    log("Using view index {} as the camera master.".format(master_idx))

    # Equalize cameras: master view's camera is copied to every other view.
    for i, v in enumerate(render_views):
        if i == master_idx:
            continue
        copy_camera(master, v)

    base_position = list(master.CameraPosition)
    base_focal = list(master.CameraFocalPoint)
    base_viewup = list(master.CameraViewUp)

    if ROTATION_CENTER is not None:
        cx, cy = ROTATION_CENTER[0], ROTATION_CENTER[1]
    else:
        cx, cy = base_focal[0], base_focal[1]
    log("Rotation axis: z-line through ({:.6g}, {:.6g}).".format(cx, cy))

    timesteps = collect_timesteps()
    log("Timesteps to render: {}".format(len(timesteps)))
    if timesteps and timesteps[0] is not None:
        log("  first={:.6g}  last={:.6g}".format(timesteps[0], timesteps[-1]))

    if VIDEO_LENGTH_SEC is not None and VIDEO_LENGTH_SEC > 0:
        frames_per_snapshot = max(1, int(round(
            VIDEO_LENGTH_SEC * FRAME_RATE / max(1, len(timesteps))
        )))
        log("VIDEO_LENGTH_SEC={}s @ {}fps over {} snapshot(s) "
            "-> frames_per_snapshot={}".format(
                VIDEO_LENGTH_SEC, FRAME_RATE, len(timesteps),
                frames_per_snapshot,
            ))
    else:
        frames_per_snapshot = FRAMES_PER_SNAPSHOT
    total_frames = len(timesteps) * frames_per_snapshot
    log("Will render {} frame(s), playback length ~{:.2f}s.".format(
        total_frames, total_frames / float(FRAME_RATE)))
    frame_idx = 0

    for ti, t in enumerate(timesteps):
        set_animation_time(t)
        for fi in range(frames_per_snapshot):
            angle = (ti + fi / float(frames_per_snapshot)) * DEG_PER_SNAPSHOT
            new_pos = rotate_point_around_z(
                base_position[0], base_position[1], base_position[2],
                cx, cy, angle,
            )
            new_vup = rotate_vec_around_z(
                base_viewup[0], base_viewup[1], base_viewup[2], angle,
            )
            set_camera_on_views(render_views, new_pos, base_focal, new_vup)

            for v in render_views:
                Render(v)

            fname = os.path.join(FRAMES_DIR, "frame_{:05d}.png".format(frame_idx))
            SaveScreenshot(fname, GetLayout(), ImageResolution=IMAGE_RESOLUTION)

            t_label = "n/a" if t is None else "{:.6g}".format(t)
            log("  [{:>4d}/{:>4d}]  t={}  angle={:+7.2f}  -> {}".format(
                frame_idx + 1, total_frames, t_label, angle,
                os.path.basename(fname),
            ))
            frame_idx += 1

    log("Rendered {} frame(s) to {}".format(frame_idx, FRAMES_DIR))
    encode_video()


def _cleanup_and_exit(rc):
    # Skip ParaView 5.9's noisy interpreter teardown (heap-corruption SIGABRT
    # during X disconnect). Kill our managed X server explicitly since
    # PDEATHSIG isn't reliable through pvbatch's threads. Wrapped in best-effort
    # try/except because pvbatch swaps sys.stdout for a stream-capture proxy.
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        os.fsync(sys.stdout.fileno())
    except (OSError, AttributeError):
        pass
    if _xserver is not None and _xserver.poll() is None:
        try:
            _xserver.terminate()
            _xserver.wait(timeout=3)
        except Exception:
            try:
                _xserver.kill()
            except Exception:
                pass
    os._exit(rc)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        _cleanup_and_exit(int(e.code) if isinstance(e.code, int) else 1)
    _cleanup_and_exit(0)
