"""Report how OpenGL resolves inside this environment.

    pixi run gl-check

Run this on any machine where the MuJoCo viewer misbehaves — a window that
appears and vanishes, or `gladLoadGL error` / `GLXBadDrawable` on stderr.

How the pieces fit together: conda gives us libglvnd, which is only the
*dispatch* layer (libGL.so.1 -> libGLX.so.0 -> libGLdispatch.so.0). Dispatch
holds no driver code. At context-creation time it asks the X server which
vendor owns the screen (the GLX_EXT_libglvnd extension) and dlopens
`libGLX_<vendor>.so.0` — which must come from the machine's own driver install
and must match the running kernel module. So a healthy setup is conda dispatch
plus a host vendor library, and the two halves are found independently.

That means the useful question isn't "which libGL.so.1 got loaded" but "which
vendor library did it dispatch to". This prints both.

Levers when it goes wrong:
  __GLX_VENDOR_LIBRARY_NAME=nvidia     force the GLX vendor (skips the X query,
                                       which is what fails under Xwayland or a
                                       remote/indirect X server)
  __EGL_VENDOR_LIBRARY_FILENAMES=...   point at a specific EGL vendor JSON
  MUJOCO_GL=egl | osmesa               make MuJoCo skip GLX entirely
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys

GL_VENDOR, GL_RENDERER, GL_VERSION = 0x1F00, 0x1F01, 0x1F02


def _loaded_gl_objects() -> list[str]:
    """Shared objects with GL in the name that this process actually mapped."""
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.rsplit(" ", 1)[-1].strip()
            base = os.path.basename(path)
            if base.startswith(("libGL", "libEGL", "libOpenGL")):
                paths.add(path)
    return sorted(paths)


def _gl_init_like_main(tag: str) -> None:
    """Replay, inside a child, exactly what the passing main-process path does.

    The plain `gl-check` run works on machines where every subprocess case fails,
    and the one thing it does that the others don't is drive GL through libglvnd
    itself — dlopen by soname and call glGetString — after the context is current
    and before MuJoCo loads glad. With libglvnd that first call is what binds the
    vendor's dispatch table for the context; if MuJoCo's glad runs before any
    such call, it can resolve against an unbound table and report
    `gladLoadGL error`.
    """
    try:
        import glfw

        if not glfw.init():
            print(f"  {tag}: glfw.init() FAILED", flush=True)
            return
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, tag, None, None)
        if not win:
            print(f"  {tag}: create_window FAILED", flush=True)
            return
        glfw.make_context_current(win)

        gl = ctypes.CDLL("libGL.so.1")
        gl.glGetString.restype = ctypes.c_char_p
        vendor = gl.glGetString(GL_VENDOR)
        print(f"  {tag}: glGetString(GL_VENDOR) -> {vendor.decode() if vendor else '<null>'}", flush=True)
        for soname in ("libGLX.so.0", "libGLdispatch.so.0", "libOpenGL.so.0", "libEGL.so.1"):
            try:
                ctypes.CDLL(soname)
            except OSError:
                pass

        import mujoco
        from robot_descriptions import panda_mj_description

        model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        print(f"  {tag}: GL init OK", flush=True)
    except BaseException as exc:                               # noqa: BLE001
        print(f"  {tag}: FAILED {type(exc).__name__}: {exc}", flush=True)


def _gl_init_in_child_ordered(tag: str, context_first: bool, mujoco_gl: str | None) -> None:
    """Same GL init, varying two things MuJoCo is sensitive to.

    ``context_first`` creates the GLFW context *before* importing mujoco.
    MuJoCo picks and initialises a GL backend when it is imported, so importing
    it into a process with no current context can bind it to a different backend
    (EGL/OSMesa) than the GLX context it is later asked to render into — glad
    then resolves an entry-point table that doesn't match, i.e. `gladLoadGL
    error`. ``mujoco_gl`` sets MUJOCO_GL to force that choice explicitly.
    """
    try:
        if mujoco_gl is not None:
            os.environ["MUJOCO_GL"] = mujoco_gl
        import glfw

        if context_first:
            if not glfw.init():
                print(f"  {tag}: glfw.init() FAILED", flush=True)
                return
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            win = glfw.create_window(64, 64, tag, None, None)
            if not win:
                print(f"  {tag}: create_window FAILED", flush=True)
                return
            glfw.make_context_current(win)

        import mujoco
        from robot_descriptions import panda_mj_description

        if not context_first:
            if not glfw.init():
                print(f"  {tag}: glfw.init() FAILED", flush=True)
                return
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            win = glfw.create_window(64, 64, tag, None, None)
            if not win:
                print(f"  {tag}: create_window FAILED", flush=True)
                return
            glfw.make_context_current(win)

        model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        print(f"  {tag}: GL init OK", flush=True)
    except BaseException as exc:                               # noqa: BLE001
        print(f"  {tag}: FAILED {type(exc).__name__}: {exc}", flush=True)


def _gl_init_in_child(tag: str, preload: bool = False) -> None:
    """Do exactly what a RobotNode does: open a window, then let MuJoCo load GL.

    With ``preload``, first pull the environment's own GL libraries into the
    process by absolute path. MuJoCo's glad resolves GL by dlopening
    ``libOpenGL.so.0`` / ``libGL.so.1`` by soname; if that lands on a different
    libglvnd than the one the context came from, the entry-point table comes
    back empty and glad reports `gladLoadGL error`. Loading the right ones first
    means glad's dlopen returns the already-mapped copy.
    """
    try:
        import mujoco
        from robot_descriptions import panda_mj_description

        import glfw

        print(f"  {tag}: LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '<unset>')}", flush=True)
        if preload:
            prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
            for soname in ("libGLdispatch.so.0", "libOpenGL.so.0", "libGLX.so.0", "libGL.so.1"):
                path = os.path.join(prefix, "lib", soname)
                if os.path.exists(path):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    print(f"  {tag}: preloaded {path}", flush=True)

        if not glfw.init():
            print(f"  {tag}: glfw.init() FAILED", flush=True)
            return
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        win = glfw.create_window(64, 64, tag, None, None)
        if not win:
            print(f"  {tag}: create_window FAILED", flush=True)
            return
        glfw.make_context_current(win)
        model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        print(f"  {tag}: GL init OK", flush=True)
    except BaseException as exc:                               # noqa: BLE001
        print(f"  {tag}: FAILED {type(exc).__name__}: {exc}", flush=True)


def check_subprocess_modes() -> int:
    """Compare GL init in a forked child vs a spawned one.

    `gl-check` on its own runs in a plain process, but session nodes run in
    subprocesses. A forked child inherits the parent's address space, and some
    drivers won't initialise GL in a process that inherited their state. If fork
    fails here and spawn succeeds, run the session with RR_START_METHOD=spawn.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")   # fork and spawn behave the same; match the session default

    # (label, context_first, MUJOCO_GL)
    cases = [
        ("baseline (mujoco imported first)", False, None),
        ("context created before importing mujoco", True, None),
        ("MUJOCO_GL=glfw", False, "glfw"),
        ("MUJOCO_GL=egl", False, "egl"),
        ("MUJOCO_GL=osmesa", False, "osmesa"),
        ("MUJOCO_GL=glfw + context first", True, "glfw"),
    ]
    for label, context_first, mujoco_gl in cases:
        print(f"case: {label}")
        proc = ctx.Process(target=_gl_init_in_child_ordered, args=(label, context_first, mujoco_gl))
        proc.start()
        proc.join(120)
        print(f"  exitcode={proc.exitcode}  (0 clean; negative = killed by that signal)")

    label = "replay of the passing main-process sequence (glGetString before mujoco)"
    print(f"case: {label}")
    proc = ctx.Process(target=_gl_init_like_main, args=(label,))
    proc.start()
    proc.join(120)
    print(f"  exitcode={proc.exitcode}  (0 clean; negative = killed by that signal)")
    return 0


def main() -> int:
    if "--subprocess" in sys.argv:
        return check_subprocess_modes()

    print("environment overrides:")
    for var in (
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "__GLX_VENDOR_LIBRARY_NAME",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "MUJOCO_GL",
        "LD_LIBRARY_PATH",
    ):
        print(f"  {var:32s} {os.environ.get(var, '<unset>')}")

    print("\nvendor libraries present on this machine:")
    vendors = sorted(glob.glob("/usr/lib/*/libGLX_*.so.0")) + sorted(glob.glob("/usr/lib/libGLX_*.so.0"))
    for v in vendors or ["  <none found — the driver's GLX vendor library is missing>"]:
        print(f"  {v}")
    egl_json = sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json"))
    for j in egl_json or ["  <no EGL vendor JSON>"]:
        print(f"  {j}")

    print("\ncreating a GL context:")
    try:
        import glfw
    except ImportError:
        print("  glfw not installed — skipping the live context test")
        return 1
    if not glfw.init():
        print("  glfw.init() FAILED — no usable display")
        return 1
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "gl-check", None, None)
    if not win:
        glfw.terminate()
        print("  create_window FAILED — GLX could not give us a context")
        return 1
    glfw.make_context_current(win)

    gl = ctypes.CDLL("libGL.so.1")
    gl.glGetString.restype = ctypes.c_char_p
    for label, enum in (("GL_VENDOR", GL_VENDOR), ("GL_RENDERER", GL_RENDERER), ("GL_VERSION", GL_VERSION)):
        raw = gl.glGetString(enum)
        print(f"  {label:12s} {raw.decode() if raw else '<null>'}")

    print("\nGL objects actually mapped (dispatch should be conda, vendor should be the host driver):")
    for path in _loaded_gl_objects():
        origin = "conda" if ".pixi/envs" in path else "host"
        print(f"  [{origin}] {path}")

    # MuJoCo's `gladLoadGL error` means it resolved a GL entry point table that
    # doesn't match the context. That happens when the dispatch libraries get
    # mixed: a dlopen resolves against the RPATH of whichever binary calls it, so
    # conda-built Python reaches the env's libglvnd while a manylinux wheel's .so
    # reaches the host's through the ld.so cache. Two libGLdispatch instances,
    # and the entry points come back null.
    print("\nwhere a plain dlopen lands for each soname (all should agree — all conda or all host):")
    for soname in ("libGL.so.1", "libGLX.so.0", "libGLdispatch.so.0", "libOpenGL.so.0", "libEGL.so.1"):
        try:
            ctypes.CDLL(soname)
        except OSError as exc:
            print(f"  {soname:22s} NOT FOUND ({exc})")
            continue
        hits = [p for p in _loaded_gl_objects() if os.path.basename(p).startswith(soname.split(".so")[0] + ".so")]
        for h in hits:
            print(f"  {soname:22s} [{'conda' if '.pixi/envs' in h else 'host'}] {h}")

    print("\nMuJoCo's own GL init (this is what prints `gladLoadGL error`):")
    try:
        import mujoco
        from robot_descriptions import panda_mj_description

        model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        print("  MjrContext created — MuJoCo can load GL in this environment")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    glfw.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
