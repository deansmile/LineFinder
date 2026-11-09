"""Write timing_summary.md (+ timings_per_image.csv) from the timing passes.

Inputs: <results>/timed/timings_per_image.csv (JointBDOE, DepthPro, RANSAC passes),
        <results>/timed/timings_describe.csv (description pass), <results>/api_timing/api_timings_*.csv,
        env_*.json files written by each pass.
Stats per stage/model: n, mean, median, standard deviation (sample), p90, in seconds.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import time


def p90(xs):
    xs = sorted(xs)
    k = 0.9 * (len(xs) - 1)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path, ok_only=False):
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    return [r for r in rows if r.get("seconds") and (not ok_only or r.get("ok", "1") == "1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    R = args.results
    pipe = load(os.path.join(R, "timed", "timings_per_image.csv")) + load(os.path.join(R, "timed", "timings_describe.csv"))
    api_rows, api_fail = [], {}
    for f in sorted(glob.glob(os.path.join(R, "api_timing", "api_timings_*.csv"))):
        rows = list(csv.DictReader(open(f)))
        api_rows += [r for r in rows if r["ok"] == "1"]
        api_fail[rows[0]["method"]] = (sum(r["ok"] != "1" for r in rows), len(rows))

    # per-image total for LineFinder: sum of the per-image stage times of each pass
    stages_for_total = ["JointBDOE (YOLOv5 detection + orientation)", "DepthPro (metric depth)",
                        "RANSAC + endpoint logic (total)", "Description step (total)"]
    by_img = {}
    for r in pipe:
        if r["stage"] in stages_for_total:
            by_img.setdefault(r["image_id"], {})[r["stage"]] = float(r["seconds"])
    totals = [sum(v.values()) for v in by_img.values() if len(v) == len(stages_for_total)]

    groups = {}
    for r in pipe + api_rows:
        groups.setdefault((r["method"], r["stage"]), []).append(float(r["seconds"]))
    if totals:
        groups[("LineFinder", "Total per image (sum of passes)")] = totals

    with open(os.path.join(R, "timings_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "stage", "image_id", "width", "height", "seconds", "note"])
        for r in pipe:
            w.writerow([r["method"], r["stage"], r["image_id"], r["width"], r["height"], r["seconds"], r.get("note", "")])
        for r in api_rows:
            w.writerow([r["method"], r["stage"], r["image_id"], r["width"], r["height"], r["seconds"],
                        "API round trip, started %s" % r["started"]])

    envs = {}
    for f in sorted(glob.glob(os.path.join(R, "timed", "env_*.json")) + glob.glob(os.path.join(R, "api_timing", "env_*.json"))):
        envs[os.path.basename(f)[4:-5]] = json.load(open(f))

    order = ["JointBDOE (YOLOv5 detection + orientation)", "DepthPro (metric depth)", "RANSAC count",
             "RANSAC start/end + visibility", "RANSAC facing", "RANSAC + endpoint logic (total)",
             "Description (GPT-5.4 mini call)", "Description step (total)", "Total per image (sum of passes)"]
    keys = sorted(groups, key=lambda k: (k[0] != "LineFinder", order.index(k[1]) if k[1] in order else 99, k[0]))
    L = ["# LineFinder timing summary", "", "Generated %s from `%s`." % (time.strftime("%Y-%m-%d %H:%M %Z"), R), "",
         "| Method | Stage | n | Mean (s) | Median (s) | Std (s) | p90 (s) |", "|---|---|---:|---:|---:|---:|---:|"]
    for k in keys:
        xs = groups[k]
        L.append("| %s | %s | %d | %.3f | %.3f | %.3f | %.3f |" % (k[0], k[1], len(xs), statistics.mean(xs),
                                                               statistics.median(xs),
                                                               statistics.stdev(xs) if len(xs) > 1 else 0.0, p90(xs)))
    L += ["", "Failed API calls (excluded from the statistics):", ""]
    for m, (n_fail, n) in api_fail.items():
        L.append("- %s: %d of %d" % (m, n_fail, n))
    L += ["", "## Hardware and software", "", "| Pass | Date | Host | GPU | CPU | CPUs allocated | RAM | OS | Python | PyTorch | CUDA (torch) |",
          "|---|---|---|---|---|---:|---|---|---|---|---|"]
    for name, e in envs.items():
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, e.get("date", ""), e.get("host", ""), e.get("torch_device", e.get("gpu", "")) if e.get("cuda_available") else "none (CPU)",
            e.get("cpu", ""), e.get("cpus_allocated", ""), e.get("ram_total", ""), e.get("os", "").strip('"'),
            e.get("python", ""), e.get("torch", "n/a"), e.get("torch_cuda", "n/a")))
    L += ["", "## How the times were measured", "",
          "- **Clock:** `time.perf_counter()` per image. GPU stages call `torch.cuda.synchronize()` before and after. Model loading is excluded; 5 untimed warm-up images run first in every LineFinder pass.",
          "- **Separate passes:** JointBDOE, DepthPro and the LineFinder logic need three different Python environments, so each is its own pass over all 427 images, timed per image within the pass. The LineFinder total is the per-image sum of the pass times.",
          "- **No separate YOLO stage:** JointBDOE is a single YOLOv5-based network that outputs person boxes and body orientation in one forward pass, so detection and orientation are one stage.",
          "- **What each stage includes:** JointBDOE and DepthPro include reading and preprocessing the image. The RANSAC stages include loading the depth and box arrays; count, start/end + visibility, and facing each fit their own RANSAC line (seed 42), as the pipeline does.",
          "- **Description step:** start/end selection, cropping, and one GPT-5.4 mini call per *visible* endpoint (0–2 calls). The call row covers only images with at least one call. It ran in its own CPU-node pass, since it is network-bound.",
          "- **Hardware:** the GPU passes ran on one NVIDIA A100-SXM4-80GB, allocated exclusively to the job, with 16 CPU cores and 128 GB of RAM allocated. A whole dedicated node was not available (the cluster rejects `--exclusive`, and a 4-GPU request was blocked by the group GPU limit), so other jobs may have shared the node's CPUs. The RAM column shows the node total.",
          "- **API timing:** 100 images drawn with seed 42 from the 427 (`api_timing/api_sample_seed42.txt`). Each benchmark script's own request code was used and only the API round trip was timed (network included). The scripts' fixed sleeps between requests were disabled, and there were no retries. These outputs are not used for accuracy.",
          "- **Kimi:** the benchmark model `kimi-k2.5` is no longer served by Moonshot, so `kimi-k2.6` was timed as a stand-in, with thinking disabled. It ran at temperature 0.6, the only value this model accepts (the benchmark used 0.2).",
          "- **Output check:** the timed passes were compared with the unmodified pipeline run (`compare_outputs.json`). Boxes, orientations, focal lengths and all LineFinder CSVs were identical. 421 of 427 depth maps were bit-identical; 6 differed by at most 5e-5 in mean relative depth (half-precision GPU nondeterminism), and LineFinder's outputs computed from those depth maps were identical, apart from a float alignment score in 3 images."]
    open(os.path.join(R, "timing_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
