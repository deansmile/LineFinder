I need per-image inference times for the LineFinder paper (reviewer request). Measure them on the existing pipeline. Don't change any model, prompt, threshold or saved result.

1. The code is in /scratch/ds5725/linefinder
   - Locate the LineFinder pipeline: YOLO person detection, DepthPro metric depth, JointBDOE body orientation, RANSAC line fitting plus the endpoint/visibility logic, and the Gemini 3 Flash call that describes the cropped endpoint persons.
   - Locate the VLM baseline scripts for Gemini 3 Flash, GPT-5.4 Mini High and Kimi-K2.5 Instant.
   - Locate the 427-image evaluation set (the images not marked invalid).
   - Show me what you found before changing anything.

2. Add timing to LineFinder in a separate script or behind a flag, not by editing the main code path
   - Load all models once before timing. Model loading is not part of per-image time.
   - Run 5 untimed warm-up images first.
   - Use time.perf_counter(). Call torch.cuda.synchronize() before and after each GPU stage.
   - Time each stage separately: YOLO, DepthPro, JointBDOE, RANSAC plus endpoint logic, Gemini call (network included), and the total per image.
   - Record each image's resolution.

3. Time each VLM baseline
   - Use the exact same prompts and settings as the benchmark.
   - Measure end-to-end API time per image, network latency included.
   - Leave out failed or retried calls, and log how many there were.
   - Record the date and time of the run.

4. Record the hardware
   - GPU model and memory (nvidia-smi), CPU model, RAM, OS, and Python, PyTorch and CUDA versions.

5. Before running, estimate the API cost of timing all 427 images for all four API-based methods. Ask me whether to run all 427 or a random subset of 100 with a fixed seed.

6. Output
   - Save timings_per_image.csv with one row per image per method and stage.
   - Save timing_summary.md with the hardware, n, and the mean, median, standard deviation and p90 for every stage and method.
   - Don't overwrite any existing result files.
   - Check that LineFinder's outputs on the timed images match the saved results, so we know the instrumentation changed nothing.

Report the summary table when done.