"""Timed start/end person description pass (GPT-5.4 mini), `annot` env + openai>=1.

Calls the unmodified per-image function of batch_describe_start_end_person_gpt5.py
(process_one_image: start/end selection, crop, one Responses API call per visible
endpoint) and writes the same CSV as that script. Only one pass is made, so these are the
official descriptions for the rerun; no second set of API calls is spent on a comparison.

Per image it records:
  - "Description (GPT-5.4 mini call)": total wall time of the API calls (network included)
  - "Description step (total)": selection + crop + encode + API calls
The client's responses.create is wrapped to measure the calls; nothing else is changed.
"""
import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Timer, TimingLog, image_size, write_env  # noqa: E402
import batch_describe_start_end_person_gpt5 as bd  # noqa: E402

FIELDS = ["source_csv", "image_id", "image_path", "start_of_line_visible", "end_of_line_visible",
          "start_description", "end_description", "status", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--input-csv", required=True, help="line_visibility_new.csv from the rerun")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--timing-csv", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--model", default="gpt-5.4-mini-2026-03-17")  # script default
    ap.add_argument("--pad", type=int, default=8)                   # script default
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    write_env(args.env_dir, "describe")
    client = bd.create_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
    calls = {"t": 0.0, "n": 0, "err": 0}
    orig = client.responses.create

    def timed_create(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        except Exception:
            calls["err"] += 1
            raise
        finally:
            calls["t"] += time.perf_counter() - t0
            calls["n"] += 1
    client.responses.create = timed_create

    sargs = SimpleNamespace(root=args.root, model=args.model, pad=args.pad)
    rows = bd.read_all_visibility_csvs([args.input_csv])
    for r in rows[:args.warmup]:
        try:
            bd.process_one_image(r, sargs, client)
        except Exception:
            pass

    log = TimingLog(args.timing_csv)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            calls.update(t=0.0, n=0, err=0)
            wd, ht = image_size(r["image_path"])
            with Timer() as tm:
                try:
                    out = bd.process_one_image(r, sargs, client)
                except Exception as e:
                    out = {k: r.get(k, "") for k in FIELDS}
                    out.update(start_description="", end_description="", status="error", error=str(e))
            note = "%d API call(s)%s" % (calls["n"], "; %d failed" % calls["err"] if calls["err"] else "")
            if calls["n"] and not calls["err"]:
                log.add("LineFinder", "Description (GPT-5.4 mini call)", r["image_id"], wd, ht, calls["t"], note)
            log.add("LineFinder", "Description step (total)", r["image_id"], wd, ht, tm.seconds, note)
            w.writerow({k: out.get(k, "") for k in FIELDS}); f.flush()
            print("[%d/%d] %s calls=%d %.2fs %s" % (i + 1, len(rows), r["image_id"], calls["n"], tm.seconds,
                                                   out.get("status")), flush=True)
    log.close()


if __name__ == "__main__":
    main()
