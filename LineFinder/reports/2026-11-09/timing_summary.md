# LineFinder timing summary

Generated 2026-11-09 05:57 EST from `/scratch/ds5725/linefinder/LineFinder/results/2026-11-09`.

| Method | Stage | n | Mean (s) | Median (s) | Std (s) | p90 (s) |
|---|---|---:|---:|---:|---:|---:|
| LineFinder | JointBDOE (YOLOv5 detection + orientation) | 427 | 0.023 | 0.019 | 0.013 | 0.034 |
| LineFinder | DepthPro (metric depth) | 427 | 0.138 | 0.130 | 0.030 | 0.147 |
| LineFinder | RANSAC count | 427 | 0.065 | 0.063 | 0.005 | 0.068 |
| LineFinder | RANSAC start/end + visibility | 427 | 0.085 | 0.080 | 0.017 | 0.095 |
| LineFinder | RANSAC facing | 427 | 0.066 | 0.065 | 0.005 | 0.070 |
| LineFinder | RANSAC + endpoint logic (total) | 427 | 0.216 | 0.210 | 0.024 | 0.231 |
| LineFinder | Description (GPT-5.4 mini call) | 249 | 1.567 | 1.300 | 0.656 | 2.517 |
| LineFinder | Description step (total) | 427 | 0.991 | 1.158 | 0.922 | 2.392 |
| LineFinder | Total per image (sum of passes) | 427 | 1.367 | 1.529 | 0.923 | 2.757 |
| GPT-5.4 mini (gpt-5.4-mini-2026-03-17) | API call (end-to-end) | 100 | 1.484 | 1.402 | 0.306 | 1.926 |
| Gemini 3 Flash (gemini-3-flash-preview) | API call (end-to-end) | 100 | 16.720 | 10.323 | 19.912 | 31.768 |
| Kimi kimi-k2.6 Instant (stand-in for kimi-k2.5; thinking disabled; temperature 0.6) | API call (end-to-end) | 100 | 8.831 | 6.834 | 3.994 | 14.736 |

Failed API calls (excluded from the statistics):

- Gemini 3 Flash (gemini-3-flash-preview): 0 of 100
- GPT-5.4 mini (gpt-5.4-mini-2026-03-17): 0 of 100
- Kimi kimi-k2.6 Instant (stand-in for kimi-k2.5; thinking disabled; temperature 0.6): 0 of 100

## Hardware and software

| Pass | Date | Host | GPU | CPU | CPUs allocated | RAM | OS | Python | PyTorch | CUDA (torch) |
|---|---|---|---|---|---:|---|---|---|---|---|
| api_gemini | 2026-11-09 05:13:35 EST | gl062.hpc.nyu.edu | none (CPU) | INTEL(R) XEON(R) PLATINUM 8592+ | 4 | 503 GiB | Ubuntu 22.04.2 LTS | 3.10.15 | not installed | n/a |
| api_gpt | 2026-11-09 04:56:02 EST | cs620.hpc.nyu.edu | none (CPU) | INTEL(R) XEON(R) PLATINUM 8592+ | 2 | 503 GiB | Ubuntu 22.04.2 LTS | 3.10.15 | not installed | n/a |
| api_kimi | 2026-11-09 05:41:34 EST | gl062.hpc.nyu.edu | none (CPU) | INTEL(R) XEON(R) PLATINUM 8592+ | 4 | 503 GiB | Ubuntu 22.04.2 LTS | 3.10.15 | not installed | n/a |
| depthpro | 2026-11-09 05:38:53 EST | ga007.hpc.nyu.edu | NVIDIA A100-SXM4-80GB | Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz | 16 | 503 GiB | Ubuntu 22.04.2 LTS | 3.9.25 | 2.8.0+cu128 | 12.8 |
| describe | 2026-11-09 05:43:14 EST | cs617.hpc.nyu.edu | none (CPU) | INTEL(R) XEON(R) PLATINUM 8592+ | 4 | 503 GiB | Ubuntu 22.04.2 LTS | 3.10.15 | not installed | n/a |
| jointbdoe | 2026-11-09 05:40:39 EST | ga007.hpc.nyu.edu | NVIDIA A100-SXM4-80GB | Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz | 16 | 503 GiB | Ubuntu 20.04.2 LTS | 3.8.20 | 2.4.1+cu121 | 12.1 |
| linefinder_logic | 2026-11-09 05:41:10 EST | ga007.hpc.nyu.edu | none (CPU) | Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz | 16 | 503 GiB | Ubuntu 22.04.2 LTS | 3.10.15 | not installed | n/a |

## How the times were measured

- **Clock:** `time.perf_counter()` per image. GPU stages call `torch.cuda.synchronize()` before and after. Model loading is excluded; 5 untimed warm-up images run first in every LineFinder pass.
- **Separate passes:** JointBDOE, DepthPro and the LineFinder logic need three different Python environments, so each is its own pass over all 427 images, timed per image within the pass. The LineFinder total is the per-image sum of the pass times.
- **No separate YOLO stage:** JointBDOE is a single YOLOv5-based network that outputs person boxes and body orientation in one forward pass, so detection and orientation are one stage.
- **What each stage includes:** JointBDOE and DepthPro include reading and preprocessing the image. The RANSAC stages include loading the depth and box arrays; count, start/end + visibility, and facing each fit their own RANSAC line (seed 42), as the pipeline does.
- **Description step:** start/end selection, cropping, and one GPT-5.4 mini call per *visible* endpoint (0–2 calls). The call row covers only images with at least one call. It ran in its own CPU-node pass, since it is network-bound.
- **Hardware:** the GPU passes ran on one NVIDIA A100-SXM4-80GB, allocated exclusively to the job, with 16 CPU cores and 128 GB of RAM allocated. A whole dedicated node was not available (the cluster rejects `--exclusive`, and a 4-GPU request was blocked by the group GPU limit), so other jobs may have shared the node's CPUs. The RAM column shows the node total.
- **API timing:** 100 images drawn with seed 42 from the 427 (`api_timing/api_sample_seed42.txt`). Each benchmark script's own request code was used and only the API round trip was timed (network included). The scripts' fixed sleeps between requests were disabled, and there were no retries. These outputs are not used for accuracy.
- **Kimi:** the benchmark model `kimi-k2.5` is no longer served by Moonshot, so `kimi-k2.6` was timed as a stand-in, with thinking disabled. It ran at temperature 0.6, the only value this model accepts (the benchmark used 0.2).
- **Output check:** the timed passes were compared with the unmodified pipeline run (`compare_outputs.json`). Boxes, orientations, focal lengths and all LineFinder CSVs were identical. 421 of 427 depth maps were bit-identical; 6 differed by at most 5e-5 in mean relative depth (half-precision GPU nondeterminism), and LineFinder's outputs computed from those depth maps were identical, apart from a float alignment score in 3 images.
