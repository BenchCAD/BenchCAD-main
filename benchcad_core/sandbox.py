"""A working directory the model can run Python in, with the target views in it.

Modelled on the setup Anthropic describes for their Vision2Code tool ablation:
a container holding the image files and standard Python libraries, plus an image
cropping tool. The decisive part is that the target views are *files on disk*,
not just pictures in the conversation — that is what lets the model measure
rather than eyeball. It can render its own solid through the same renderer that
produced the target, crop either image to inspect a chamfer or a hole, and
compare them numerically before committing to an answer.

Isolation: model-written code runs in a container, unprivileged, with no
network, no environment, capped memory and CPU, and exactly one writable path —
the record's own working directory. Nothing it does can reach the host
filesystem, the repository, or any credential.

`DOCKER_IMAGE` must exist (see docker/sandbox.Dockerfile); it pins the same
CadQuery stack the harness scores with, so a program that runs here runs
identically when re-executed for scoring. If the image is missing the sandbox
refuses to run rather than silently falling back to a bare subprocess — a
degraded sandbox that still executes model code is worse than none, because it
looks safe.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Native to the host architecture. An amd64 image under QEMU on Apple silicon
# was heavy enough that three concurrent records took the machine down; the
# emulation, not the container, was the cost.
DOCKER_IMAGE = "benchcad-sandbox:arm64"
MEMORY = "2g"
CPUS = "1"

# Helper module dropped into the working directory. `render` deliberately uses
# the harness renderer, so the model's picture of its own part is directly
# comparable with the target rather than merely similar.
TOOLS_PY = '''\
"""Tools available in this working directory."""
from pathlib import Path

import os, shutil, tempfile

os.environ.setdefault("BENCH_MESH_COLOR", "129,216,208")
import _render

# Composite width of the target this record must match. The renderer's own
# default is the lite set's 128-per-view; the public set's targets are rendered
# at twice that, so a hardcoded default silently produces an image half the
# target's size and every numeric comparison the model attempts dies on a shape
# mismatch. It did, for 95% of records in a 6-hour run.
TARGET_W = {target_w}
VIEW_SIZE = TARGET_W // 2 - 2 * _render.BORDER


def render(step_path, out_png="my_render.png"):
    """Render a STEP to the 2x2 composite, pixel-identical in convention to the
    target image: same cameras, same size, same colour, so the two can be
    compared directly. Returns the output path. Also writes <stem>_v0..v3.png
    for the individual views."""
    out = Path(out_png)
    tmp = tempfile.mkdtemp(dir=".")
    _render.render_step_normalized(str(Path(step_path).resolve()), tmp,
                                   size=VIEW_SIZE)
    shutil.move(f"{tmp}/composite.png", out)
    for i in range(4):
        src = Path(tmp) / f"view_{i}.png"
        if src.exists():
            shutil.move(str(src), f"{out.stem}_v{i}.png")
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def views(image_path="target.png"):
    """Split a 2x2 composite into its four quadrants; returns their paths."""
    from PIL import Image
    im = Image.open(image_path); w, h = im.size
    outs = []
    for i, box in enumerate([(0,0,w//2,h//2), (w//2,0,w,h//2),
                             (0,h//2,w//2,h), (w//2,h//2,w,h)]):
        o = Path(f"{Path(image_path).stem}_v{i}.png"); im.crop(box).save(o); outs.append(o)
    return outs


def crop(image_path, box, out_png=None):
    """Crop `image_path` to box=(left, top, right, bottom) and save it.

    The composite is a 2x2 grid, so a single view is one quadrant — here the
    top-left view is box=(0, 0, TARGET_W // 2, TARGET_W // 2)."""
    from PIL import Image
    im = Image.open(image_path)
    out = Path(out_png or f"crop_{Path(image_path).stem}.png")
    im.crop(tuple(box)).save(out)
    return out


def export(result, step_path="my_part.step"):
    """Export a CadQuery Workplane or Shape to STEP."""
    out = Path(step_path)
    (result.val() if hasattr(result, "val") else result).exportStep(str(out))
    return out
'''


@dataclass
class Result:
    """One execution: what the model's code printed, and any images it made."""
    returncode: int
    stdout: str
    stderr: str
    images: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _image_present() -> bool:
    r = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                       capture_output=True)
    return r.returncode == 0


def _mount_works(work_dir: Path) -> bool:
    """Whether a bind mount of `work_dir` actually carries its contents.

    Docker does not fail when asked to mount a path its VM cannot reach — it
    creates an empty directory at that path inside the VM and mounts *that*.
    The container then starts normally and every program reports a missing
    file, which is indistinguishable from the model writing bad code.

    That failure has now cost two runs. The first was macOS's `/tmp` symlink;
    the second was colima, which mounts only `$HOME` — a work directory under
    `/tmp` therefore mounts empty even after resolving the symlink. A 0.2s
    probe per record buys certainty that the sandbox is real.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "--network", "none",
         "--read-only", "--security-opt", "no-new-privileges",
         "-v", f"{work_dir}:/work", "-w", "/work",
         DOCKER_IMAGE, "python", "-c", "import pathlib,sys;"
         "sys.exit(0 if pathlib.Path('/work/tools.py').exists() else 1)"],
        capture_output=True, timeout=120)
    return r.returncode == 0


class Sandbox:
    """A per-record working directory the model executes Python in."""

    def __init__(self, work_dir: Path, target_images: dict[str, Path], repo_root: Path):
        self.dir = Path(work_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        if not _image_present():
            raise RuntimeError(
                f"sandbox image {DOCKER_IMAGE!r} not found — build it with\n"
                f"  docker build -f docker/sandbox.arm64.Dockerfile "
                f"-t {DOCKER_IMAGE} .")
        # The renderer is copied in rather than mounted from the repo: the
        # container must not be able to see the repository at all.
        (self.dir / "_render.py").write_text(
            (Path(repo_root) / "benchcad_core/scoring/lite_renderer.py").read_text())
        self.targets = {}
        for name, src in target_images.items():
            dst = self.dir / name
            shutil.copy(src, dst)
            self.targets[name] = dst

        # Size the render tool to this record's target. Substituted rather than
        # formatted because TOOLS_PY contains f-strings of its own.
        from PIL import Image
        primary = self.targets.get("target.png") or next(iter(self.targets.values()))
        (self.dir / "tools.py").write_text(
            TOOLS_PY.replace("{target_w}", str(Image.open(primary).size[0])))
        self._seeded = {p.name for p in self.dir.iterdir()}
        # Probe after seeding, so the check is that the *real* contents arrive.
        if not _mount_works(self.dir):
            raise RuntimeError(
                f"the sandbox mount of {self.dir} arrives empty inside the "
                f"container, so nothing written here is visible to the model.\n"
                f"The container runtime can only bind paths its VM shares: "
                f"colima shares $HOME only, and Docker Desktop shares a "
                f"configured list. Put the work directory under $HOME.")
        self.log_dir = self.dir.parent / (self.dir.name + "_log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._round = 0
        self._seen = self._snapshot()

    def reset(self) -> None:
        """Return the directory to its seeded state.

        Called once per record, not per round. Resetting every round was the
        original design and it was wrong: it forces each block to rebuild
        everything it needs from scratch, so the model cannot keep a helper
        script, a measurement, or the solid it built last round. The episode
        stops being a working session and becomes ten independent attempts that
        happen to share a transcript. Logs live outside the mount either way.
        """
        for p in self.dir.iterdir():
            if p.name in self._seeded:
                continue
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        self._seen = self._snapshot()

    def _snapshot(self) -> dict:
        return {p.name: p.stat().st_mtime_ns
                for p in self.dir.glob("*.png") if p.is_file()}

    def run(self, code: str, timeout: int = 300) -> Result:
        """Execute `code` in the container; report what it produced.

        Images the code writes are returned so they can be shown back to the
        model — a cropping or rendering tool the model cannot see the output of
        is not a tool.
        """
        # The scorer prepends this shim to every program it executes, so code
        # that omits it fails here while the identical code succeeds at scoring
        # -- selectors (.faces/.edges/.vertices) all raise without it. The
        # sandbox is only worth having if the two environments agree, so the
        # constant is imported rather than copied.
        from .scoring.exec_cq import _OCP_HASHCODE_FIX
        script = self.dir / "_run.py"
        script.write_text(_OCP_HASHCODE_FIX + "\n" + code)
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",              # no exfiltration, no fetching
            "--memory", MEMORY, "--cpus", CPUS,
            "--pids-limit", "256",
            "--read-only",                    # rootfs immutable
            "--tmpfs", "/tmp:size=256m",
            "--security-opt", "no-new-privileges",
            "--env-file", "/dev/null",        # no host environment at all
            "-e", "BENCH_MESH_COLOR=129,216,208",   # match the target's palette
            "-v", f"{self.dir}:/work",        # the one writable path
            "-w", "/work",
            DOCKER_IMAGE, "python", "_run.py",
        ]
        try:
            r = subprocess.run(cmd, timeout=timeout, capture_output=True)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = -1, b"", f"timeout after {timeout}s".encode()
        now = self._snapshot()
        fresh = [self.dir / n for n, m in now.items()
                 if self._seen.get(n) != m]
        self._seen = now
        res = Result(rc,
                     out.decode(errors="replace")[-4000:],
                     err.decode(errors="replace")[-4000:],
                     sorted(fresh))
        # Hand back the archived copies, not the working-directory originals.
        # The caller resets between rounds, and reset runs before the images are
        # attached to the next turn -- pointing at the originals meant every
        # render the model produced was deleted before it could be shown, so the
        # round died reading a missing file and the sandbox's whole purpose,
        # letting the model look at its own geometry, silently never happened.
        res.images = self._log(code, res)
        return res

    def _log(self, code: str, res: Result) -> list:
        """Persist the round outside the sandbox, so a reset cannot erase it and
        a failure can be diagnosed without re-running the model.

        Returns the archived image paths, which are what the caller should show
        the model — they outlive the reset.
        """
        self._round += 1
        d = self.log_dir / f"round_{self._round:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "code.py").write_text(code)
        (d / "stdout.txt").write_text(res.stdout)
        (d / "stderr.txt").write_text(res.stderr)
        (d / "returncode.txt").write_text(str(res.returncode))
        # Geometry is preserved alongside the images because the reset that runs
        # after this wipes it. Without this the runner's "fall back to the last
        # STEP the model built" is unreachable — a record whose final program
        # fails to execute scores zero even when an earlier round produced a
        # sound solid.
        kept = []
        for f in list(res.images) + sorted(self.dir.glob("*.step")):
            try:
                shutil.copy(f, d / f.name)
                if f.suffix == ".png":
                    kept.append(d / f.name)
            except OSError:
                pass
        return kept

    def artifacts(self, suffix: str) -> list:
        """Every file of `suffix` any round produced, oldest first.

        Reads the log rather than the working directory, which reset empties.
        """
        return sorted(self.log_dir.glob(f"round_*/*{suffix}"),
                      key=lambda p: (p.parent.name, p.name))
