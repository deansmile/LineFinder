"""End-to-end API latency per image for the VLM baselines, with the exact benchmark requests.

The benchmark scripts (gpt_line_reduced_questions.py, gemini_line.py, kimi_line.py) build the
prompt and send the request inside generate_reranking(). This script runs each script's own
setup code (imports, client, prompt builder, generate_reranking), stopping before its
benchmark loop, and calls generate_reranking([one_image], out) per image. Unchanged: model,
prompt, image encoding, parameters. Changed only for timing: the API method is wrapped to
measure the request round trip (time.perf_counter, network included), and the scripts' fixed
sleep() between requests is disabled. No retries; a raised API error is a failed call, logged
and excluded from the timing.

Outputs are for timing only (not used for accuracy).
  python time_vlm_api.py --model gpt|gemini|kimi --list image_list_427.txt --n 100 --seed 42 --out-dir DIR
GPT needs openai 0.28 (benchmark uses openai.ChatCompletion); Kimi needs openai>=1.
"""
import argparse
import csv
import os
import random
import sys
import time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import image_size, read_list, write_env  # noqa: E402

SCRIPTS = {"gpt": "gpt_line_reduced_questions.py", "gemini": "gemini_line.py", "kimi": "kimi_line.py"}
LOOP_MARKERS = ('root = lf_path("Images")', 'if __name__ == "__main__":')
LABEL = {"gpt": "GPT-5.4 mini (gpt-5.4-mini-2026-03-17)", "gemini": "Gemini 3 Flash (gemini-3-flash-preview)",
         "kimi": "Kimi K2.5 Instant (kimi-k2.5, thinking disabled)"}


def load_setup(model):
    path = os.path.join(CODE, SCRIPTS[model])
    src = open(path, encoding="utf-8").read()
    cut = min(src.index(m) for m in LOOP_MARKERS if m in src)
    ns = {"__file__": path, "__name__": "benchmark_setup"}
    exec(compile(src[:cut], path, "exec"), ns)
    return ns


class Clock:
    def __init__(self):
        self.last = None
        self.error = None

    def wrap(self, fn):
        def timed(*a, **kw):
            self.error = None
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            except Exception as e:
                self.error = repr(e)
                raise
            finally:
                self.last = time.perf_counter() - t0
        return timed


OVERRIDES = {}


def install(ns, model, clock):
    ns["sleep"] = lambda *_: None  # scripts pause between requests; not part of API time
    if model == "gpt":
        import openai
        openai.ChatCompletion.create = clock.wrap(openai.ChatCompletion.create)
    elif model == "kimi":
        c = ns["client"].chat.completions
        orig = c.create

        def create(*a, **kw):  # stand-in model: benchmark kimi-k2.5 is no longer served
            if OVERRIDES.get("model"):
                kw["model"] = OVERRIDES["model"]
            if OVERRIDES.get("temperature") is not None:
                kw["temperature"] = OVERRIDES["temperature"]
            return orig(*a, **kw)
        c.create = clock.wrap(create)
    else:
        m = ns["client"].models
        m.generate_content = clock.wrap(m.generate_content)
        if "tqdm" in ns:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SCRIPTS))
    ap.add_argument("--list", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kimi-model", default=None, help="stand-in for kimi-k2.5 (e.g. kimi-k2.6)")
    ap.add_argument("--kimi-temperature", type=float, default=None, help="only if the stand-in requires it")
    args = ap.parse_args()
    if args.model == "kimi":
        OVERRIDES.update(model=args.kimi_model, temperature=args.kimi_temperature)
        if args.kimi_model:
            LABEL["kimi"] = "Kimi %s Instant (stand-in for kimi-k2.5; thinking disabled; temperature %s)" % (
                args.kimi_model, args.kimi_temperature if args.kimi_temperature is not None else "0.2")

    os.makedirs(args.out_dir, exist_ok=True)
    write_env(args.out_dir, "api_" + args.model)
    paths = sorted(read_list(args.list))
    sample = random.Random(args.seed).sample(paths, args.n)
    with open(os.path.join(args.out_dir, "api_sample_seed%d.txt" % args.seed), "w") as f:
        f.write("\n".join(sample) + "\n")

    ns = load_setup(args.model)
    clock = Clock()
    install(ns, args.model, clock)
    raw_out = os.path.join(args.out_dir, "api_outputs_%s.jsonl" % args.model)
    open(raw_out, "w").close()
    tmp = raw_out + ".one"

    rows, failed = [], 0
    for i, p in enumerate(sample):
        image_id = os.path.splitext(os.path.basename(p))[0]
        w, h = image_size(p)
        clock.last, clock.error = None, None
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        open(tmp, "w").close()  # kimi_line.py appends to its output file; start each image empty
        try:
            ns["generate_reranking"]([p], tmp)
        except Exception as e:  # GPT script does not catch API errors itself
            clock.error = clock.error or repr(e)
        if clock.error is None and os.path.exists(tmp):
            txt = open(tmp, encoding="utf-8").read()
            if '"error"' in txt and "invalid_json" not in txt and len(txt) < 2000:
                clock.error = "script-caught error: " + txt.strip()[:200]
            with open(raw_out, "a", encoding="utf-8") as f:
                f.write(txt)
        ok = clock.error is None and clock.last is not None
        failed += not ok
        rows.append(dict(method=LABEL[args.model], stage="API call (end-to-end)", image_id=image_id, width=w,
                         height=h, seconds="%.6f" % clock.last if clock.last is not None else "",
                         ok=int(ok), error=clock.error or "", started=started))
        print("[%d/%d] %s %s %s" % (i + 1, len(sample), image_id, rows[-1]["seconds"], "OK" if ok else clock.error),
              flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(os.path.join(args.out_dir, "api_timings_%s.csv" % args.model), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("done: %d ok, %d failed" % (len(rows) - failed, failed))


if __name__ == "__main__":
    main()
