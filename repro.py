#!/usr/bin/env python3
# ============================================================
# repro.py — cross-platform reproduction driver (Windows + Linux)
# ------------------------------------------------------------
# One tool, four subcommands. No bash / grep needed; only Python + uv + git.
#
#   python repro.py setup      # build the env with uv (auto-detects GPU stack)
#   python repro.py patch      # apply the two quality patches (idempotent)
#   python repro.py check      # GPU / weights / minimal-eval self-check
#   python repro.py eval       # reproduce all evaluable scenarios -> results txt
#
# Run every command from the ROOT of the cloned repo (where onpolicy/ lives).
# Prereqs: `git clone` already done + `uv` installed (https://astral.sh/uv).
# ============================================================
import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_MARKER = Path("onpolicy/scripts/eval_mpe.py")

# runtime deps for the Blackwell path (versions matching the validated env)
RUNTIME_DEPS = [
    "numpy==1.26.4", "gym==0.26.2", "gymnasium==1.3.0", "pyglet==1.5.26",
    "networkx==3.6.1", "scipy==1.13.1", "wandb==0.28.0",
    "tensorboard==2.17.1", "tensorboardX==2.6.1",
    "setproctitle==1.3.7", "absl-py==2.4.0", "imageio==2.31.1",
    "tqdm==4.66.2", "PyYAML==6.0.3",
]

# (tag, scenario_name, num_agents, world_size, episode_length)
SCENARIOS = [
    ("single_N3",  "working_three_phase_graph",     3,  5, 120),
    ("single_N5",  "working_three_phase_graph",     5,  5, 120),
    ("single_N10", "working_three_phase_graph",     10, 5, 120),
    ("seq_N3",     "three_phase_graph_sequential",  3,  5, 200),
    ("seq_N5",     "three_phase_graph_sequential",  5,  5, 200),
    ("seq_N10",    "three_phase_graph_sequential",  10, 5, 200),
    ("merge_N3",   "three_phase_graph_merge",       3,  5, 200),
    ("merge_N5",   "three_phase_graph_merge",       5,  5, 200),
    ("merge_N10",  "three_phase_graph_merge",       10, 5, 200),
]

METRIC_KEYS = ("Success rates mean", "Conformance C% Mean",
               "Total Time Taken Median", "Num collisions")


# ---------- helpers ----------
def die(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)


def ensure_repo_root():
    if not REPO_MARKER.exists():
        die("run from the repo root (onpolicy/scripts/eval_mpe.py not found)")


def ensure_uv():
    if shutil.which("uv") is None:
        die("uv not installed. Install:  https://astral.sh/uv  "
            "(curl -LsSf https://astral.sh/uv/install.sh | sh)")


def venv_python() -> Path:
    """Location of the venv interpreter differs by OS."""
    if os.name == "nt":
        return Path(".venv") / "Scripts" / "python.exe"
    return Path(".venv") / "bin" / "python"


def need_venv() -> str:
    py = venv_python()
    if not py.exists():
        die(f"venv not found at {py} -- run  python repro.py setup  first")
    return str(py)


def run(cmd, **kw):
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def gpu_compute_cap():
    """Return the first GPU's compute capability as a float, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            line = line.strip()
            if line:
                return float(line)
    except Exception:
        return None
    return None


def verify_gpu(py):
    run([py, "-c",
         "import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
         "print('torch', torch.__version__, '| device', torch.cuda.get_device_name(), "
         "'| cap', torch.cuda.get_device_capability())"])


def eval_cmd(py, tag, scenario, n, world, eplen, episodes, ifi="0.0"):
    return [
        py, "-u", "onpolicy/scripts/eval_mpe.py", "--model_dir", "model_weights",
        "--scenario_name", scenario, "--dynamics_type", "air_taxi",
        "--num_agents", n, "--num_landmarks", n,
        "--world_size", world, "--episode_length", eplen,
        "--formation_type", "point", "--total_actions", "9", "--zeroshift", "10",
        "--render_episodes", episodes, "--use_dones", "False",
        "--ifi", ifi, "--model_name", tag,
    ]


# ---------- subcommand: setup ----------
def cmd_setup(args):
    ensure_repo_root()
    ensure_uv()

    stack = args.stack
    if stack == "auto":
        cap = gpu_compute_cap()
        if cap is None:
            die("could not detect GPU compute capability via nvidia-smi; "
                "pass --stack blackwell  or  --stack default  explicitly")
        stack = "blackwell" if cap >= 12.0 else "default"
        print(f">>> detected compute capability {cap} -> stack: {stack}")

    print(">>> [1/N] Create .venv (Python 3.11)")
    run(["uv", "venv", "--python", "3.11", ".venv"])
    py = str(venv_python())

    if stack == "blackwell":
        print(">>> Install torch 2.11.0 (cu128, supports Blackwell sm_120)")
        run(["uv", "pip", "install", "--python", py, "torch==2.11.0",
             "--index-url", "https://download.pytorch.org/whl/cu128"])
        print(">>> Install PyG compiled deps (prebuilt wheels, pt2.11+cu128)")
        run(["uv", "pip", "install", "--python", py,
             "torch-scatter==2.1.2", "torch-sparse==0.6.18",
             "-f", "https://data.pyg.org/whl/torch-2.11.0+cu128.html"])
        run(["uv", "pip", "install", "--python", py, "torch-geometric==2.3.1"])
        print(">>> Install remaining runtime deps")
        run(["uv", "pip", "install", "--python", py, *RUNTIME_DEPS])
    else:  # default (pre-Blackwell): repo's pinned torch 2.0.1 stack via uv
        if not Path("requirements.txt").exists():
            die("requirements.txt not found (needed for the default stack)")
        print(">>> Install repo requirements (torch 2.0.1) + PyG wheel index")
        run(["uv", "pip", "install", "--python", py,
             "-r", "requirements.txt", "-f", args.pyg_index])
        print(">>> Add gymnasium (imported by the code but missing from requirements.txt)")
        run(["uv", "pip", "install", "--python", py, "gymnasium==1.3.0"])

    print("\n>>> Verify GPU visibility")
    verify_gpu(py)
    print("\nOK: environment ready. Next:  python repro.py patch")


# ---------- subcommand: patch ----------
def _patch_file(path, old, new, marker):
    s = open(path, encoding="utf-8").read()  # text mode normalizes CRLF/LF -> \n
    if marker in s:
        print(f"  {path}: patch already present, skipping")
        return
    if old not in s:
        die(f"patch target not found in {path} -- upstream may have changed; "
            "check manually")
    # write back; text mode restores the platform line ending on write
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print(f"  {path}: patched")


def cmd_patch(args):
    ensure_repo_root()

    # Patch A: Conformance C%
    f = "onpolicy/runner/shared/graph_mpe_runner.py"
    old_a = ('\t\tprint("Conformance_percentage Mean:", conformance_percentage_mean)\n'
             '\t\tprint("Conformance_percentage median:", conformance_percentage_median)\n')
    new_a = ('\t\t# raw = violation fraction (steps an agent left the corridor after entering).\n'
             '\t\t# Paper C% = 1 - violation, higher is better.\n'
             '\t\tprint("Conformance_percentage Mean (violation frac, raw):", conformance_percentage_mean)\n'
             '\t\tprint("Conformance_percentage median (violation frac, raw):", conformance_percentage_median)\n'
             '\t\tprint("Conformance C%% Mean (higher=better):  %.2f%%" % ((1.0 - conformance_percentage_mean) * 100.0))\n'
             '\t\tprint("Conformance C%% median (higher=better): %.2f%%" % ((1.0 - conformance_percentage_median) * 100.0))\n')
    _patch_file(f, old_a, new_a, marker="higher=better")

    # Patch B: vsync=False
    f = "multiagent/rendering.py"
    old_b = "self.window = pyglet.window.Window(width=width, height=height, display=display)"
    new_b = "self.window = pyglet.window.Window(width=width, height=height, display=display, vsync=False)"
    _patch_file(f, old_b, new_b, marker="vsync=False")

    print("\nOK: patches done. Next:  python repro.py check")


# ---------- subcommand: check ----------
def cmd_check(args):
    ensure_repo_root()
    py = need_venv()

    print(">>> [1/3] GPU / torch")
    verify_gpu(py)

    print(">>> [2/3] Pretrained weights")
    for f in ("model_weights/actor.pt", "model_weights/critic.pt", "model_weights/config.yaml"):
        if Path(f).exists():
            print("  ok ", f)
        else:
            die(f"missing {f}")

    print(">>> [3/3] End-to-end minimal eval (Single, 2 episodes)")
    cmd = eval_cmd(py, "sanity", "working_three_phase_graph", "5", "5", "120", "2")
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    hits = [ln for ln in res.stdout.splitlines()
            if any(k in ln for k in ("Success rates mean", "Conformance C%", "Num collisions"))]
    if not hits:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:], file=sys.stderr)
        die("no metric lines captured -- eval likely errored (see output above)")
    for ln in hits:
        print("  ", ln)
    print("\nOK: sanity check passed. Next:  python repro.py eval")


# ---------- subcommand: eval ----------
def cmd_eval(args):
    ensure_repo_root()
    py = need_venv()
    episodes = str(args.episodes)
    ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    out_path = Path(f"repro_results_{ts}.txt")

    header = (f"# Reproduction results  {datetime.datetime.now()}\n"
              f"# EPISODES={episodes}  weights=model_weights/ (obs16, single-corridor N5 training)\n")
    print(header)
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(header + "\n")
        for tag, scenario, n, world, eplen in SCENARIOS:
            title = f"===== {tag}  (scenario={scenario} N={n} ep={eplen} episodes={episodes}) ====="
            print(title, flush=True)
            fout.write(title + "\n")
            cmd = eval_cmd(py, tag, scenario, str(n), str(world), str(eplen), episodes)
            res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
            hits = [ln for ln in res.stdout.splitlines()
                    if any(k in ln for k in METRIC_KEYS)]
            if not hits:
                msg = "  (!! no metrics captured for this run; re-run it standalone to debug)"
                print(msg)
                fout.write(msg + "\n")
                # keep the tail of raw output for debugging
                fout.write(res.stdout[-1500:] + "\n" + res.stderr[-800:] + "\n")
            else:
                for ln in hits:
                    print("  ", ln)
                    fout.write(ln + "\n")
            print(flush=True)
            fout.write("\n")

    print(f"OK: all done. Summary saved to: {out_path}")
    print("   Compare against the reference values in REPRODUCTION_GUIDE.md section 6.")


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(
        description="Cross-platform reproduction driver (Windows + Linux).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="build the env with uv")
    s.add_argument("--stack", choices=["auto", "blackwell", "default"], default="auto",
                   help="which torch stack (auto = detect via nvidia-smi)")
    s.add_argument("--pyg-index",
                   default="https://data.pyg.org/whl/torch-2.0.1+cu117.html",
                   help="PyG wheel index for the DEFAULT stack (match your CUDA)")
    s.set_defaults(func=cmd_setup)

    sub.add_parser("patch", help="apply the two quality patches").set_defaults(func=cmd_patch)
    sub.add_parser("check", help="GPU / weights / minimal-eval self-check").set_defaults(func=cmd_check)

    e = sub.add_parser("eval", help="reproduce all evaluable scenarios")
    e.add_argument("--episodes", type=int, default=100,
                   help="episodes per scenario (default 100; use 30 for a quick smoke test)")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
