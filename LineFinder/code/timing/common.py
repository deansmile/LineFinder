"""Shared helpers for the LineFinder timing harness (Python 3.8 compatible).

Each stage runs in its own environment (JointBDOE: bdoe, DepthPro: depth-pro,
LineFinder logic: annot), so every stage is a separate pass over the images.
Times are per image within each pass, measured with time.perf_counter().
"""
import csv
import json
import os
import platform
import subprocess
import sys
import time


def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def image_size(path):
    """(width, height) without decoding the full image."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


class Timer:
    """perf_counter timer; if cuda=True, synchronizes before start and before stop."""

    def __init__(self, cuda=False):
        self.cuda = cuda
        self.seconds = None

    def _sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self._t0
        return False


class TimingLog:
    """Appends rows: method, stage, image_id, width, height, seconds, pass_note."""

    FIELDS = ["method", "stage", "image_id", "width", "height", "seconds", "note"]

    def __init__(self, path):
        self.path = path
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()

    def add(self, method, stage, image_id, width, height, seconds, note=""):
        self.w.writerow(dict(method=method, stage=stage, image_id=image_id, width=width,
                             height=height, seconds="%.6f" % seconds, note=note))
        self.f.flush()

    def close(self):
        self.f.close()


def _run(cmd):
    try:
        return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=60).stdout.strip()
    except Exception as e:  # pragma: no cover
        return "unavailable: %s" % e


def environment_info(stage):
    info = {
        "stage": stage,
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "host": platform.node(),
        "os": _run("grep PRETTY_NAME /etc/os-release | cut -d= -f2"),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "cpu": _run("lscpu | grep 'Model name' | head -1 | sed 's/.*: *//'"),
        "cpus_allocated": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "ram_total": _run("free -g | awk '/Mem:/ {print $2\" GiB\"}'"),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "gpu": _run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i 0"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        pass
    return info


def write_env(out_dir, stage):
    os.makedirs(out_dir, exist_ok=True)
    info = environment_info(stage)
    with open(os.path.join(out_dir, "env_%s.json" % stage), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return info
