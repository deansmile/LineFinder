"""Table 2 (start/end person description) for all four methods on one GT, with saved judgments.

Judge: Gemini 3 Flash, through the benchmark scripts' own functions:
  strict  = eval_start_end_person_gpt_new.gemini_same_person (VLM scripts: "be conservative",
            temperature 0)
  lenient = eval_start_end_person.gemini_same_person (LineFinder script: "true if likely",
            default temperature)
Rules:
  presence   (VLM scripts): both descriptions missing -> correct; one missing -> wrong;
             both present -> judge.
  visibility (LineFinder script): both endpoints not visible -> correct; visibility disagrees
             -> wrong; otherwise the presence rule.
Scorings reported:
  benchmark : VLMs = presence + strict; LineFinder = visibility + lenient (as the paper)
  unified   : all four methods = presence + strict
Every judgment is cached in --cache (JSONL) and reused, so reruns cost nothing extra.
A failed judge call counts as wrong (as in the scripts) and is reported.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import eval_start_end_person as lf_judge          # noqa: E402  lenient prompt
import eval_start_end_person_gpt_new as vlm_judge  # noqa: E402  strict prompt
import eval_start_end_person_gpt as old_loader     # noqa: E402  robust JSON-block reader
from eval_table1_all import load_gt                # noqa: E402

DESC_KEYS = {"start": ["Description of the person at the start of the line", "start_person_description"],
             "end": ["Description of the person at the end of the line", "end_person_description"]}
VIS_KEYS = {"start": ["Do we see the start of line?", "start_visible", "start_of_line_visible"],
            "end": ["Do we see the end of line?", "end_visible", "end_of_line_visible"]}


def first(d, keys, default=""):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def yes(x):
    return str(x).strip().lower() in {"yes", "true", "1"}


def missing(desc):
    return lf_judge.is_missing_description(str(desc).strip() if desc is not None else "")


def gt_sides(group, obj):
    if group == "lo":
        a = obj["annotation"]
        return {s: (yes(first(a, VIS_KEYS[s])), str(first(a, DESC_KEYS[s])).strip()) for s in ("start", "end")}
    ppl, sol, eol = obj.get("people", {}), obj.get("start_of_line", {}), obj.get("end_of_line", {})
    return {"start": (yes(sol.get("visible", "no")), str(ppl.get("start_person_description", "")).strip()),
            "end": (yes(eol.get("visible", "no")), str(ppl.get("end_person_description", "")).strip())}


def load_vlm(files):
    preds = {}
    for f in files:
        for name, pred in old_loader.load_prediction_jsonl(f):
            preds[os.path.splitext(os.path.basename(name))[0]] = pred
    return {k: {s: (yes(first(p, VIS_KEYS[s])), str(first(p, DESC_KEYS[s])).strip()) for s in ("start", "end")}
            for k, p in preds.items()}


def load_lf(desc_csv, vis_csv):
    vis = {r["image_id"]: r for r in csv.DictReader(open(vis_csv))}
    out = {}
    for r in csv.DictReader(open(desc_csv)):
        v = vis.get(r["image_id"], r)
        out[r["image_id"]] = {"start": (yes(v.get("start_of_line_visible", "")), (r.get("start_description") or "").strip()),
                              "end": (yes(v.get("end_of_line_visible", "")), (r.get("end_description") or "").strip())}
    return out


class Judge:
    def __init__(self, cache_path):
        self.path, self.cache = cache_path, {}
        if os.path.exists(cache_path):
            for l in open(cache_path):
                if l.strip():
                    d = json.loads(l)
                    if d.get("error") is None:  # failed calls are retried on the next run
                        self.cache[d["key"]] = d
        self.f = open(cache_path, "a")
        self.calls = self.errors = 0

    @staticmethod
    def key(prompt, pred, gt):
        return hashlib.sha256(json.dumps([prompt, pred, gt]).encode()).hexdigest()

    def _call(self, prompt, pred, gt):
        fn = vlm_judge.gemini_same_person if prompt == "strict" else lf_judge.gemini_same_person
        for attempt in range(3):  # transient API errors only; the answer itself is not retried
            try:
                return bool(fn(pred, gt)), None
            except Exception as e:
                err = str(e)
                time.sleep(2 * (attempt + 1))
        return None, err

    def run(self, jobs):
        todo = [(p, a, b) for p, a, b in set(jobs) if self.key(p, a, b) not in self.cache]
        print(f"judge: {len(set(jobs))} unique pairs, {len(todo)} new calls", flush=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            for (p, a, b), (ans, err) in zip(todo, ex.map(lambda j: self._call(*j), todo)):
                rec = {"key": self.key(p, a, b), "prompt": p, "pred": a, "gt": b, "same_person": ans, "error": err,
                       "model": vlm_judge.MODEL_ID, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                self.cache[rec["key"]] = rec
                self.f.write(json.dumps(rec) + "\n"); self.f.flush()
                self.calls += 1; self.errors += err is not None

    def get(self, p, a, b):
        return self.cache[self.key(p, a, b)]["same_person"]


def side_outcome(rule, pred, gt):
    """Returns ('correct'|'wrong'|'judge', pair) for one endpoint."""
    (pv, pd), (gv, gd) = pred, gt
    if rule == "visibility":
        if not gv and not pv:
            return "correct"
        if gv != pv:
            return "wrong"
    pm, gm = missing(pd), missing(gd)
    if pm and gm:
        return "correct"
    if pm != gm:
        return "wrong"
    return "judge"


def score(preds, gt, rule, prompt, judge, collect=None):
    res = {"n_images": 0, "missing_prediction": [], "start": 0, "end": 0, "judged": 0, "judge_errors": 0}
    for iid, (group, obj) in gt.items():
        if iid not in preds:
            res["missing_prediction"].append(iid)
            pred = {"start": (False, ""), "end": (False, "")}  # no prediction -> counted as no description
        else:
            pred = preds[iid]
        res["n_images"] += 1
        g = gt_sides(group, obj)
        for s in ("start", "end"):
            out = side_outcome(rule, pred[s], g[s])
            if out == "judge":
                if collect is not None:
                    collect.append((prompt, pred[s][1], g[s][1])); continue
                ans = judge.get(prompt, pred[s][1], g[s][1]); res["judged"] += 1
                res["judge_errors"] += ans is None
                res[s] += bool(ans)
            else:
                res[s] += out == "correct"
    n = res["n_images"]
    res.update(start_acc=res["start"] / n, end_acc=res["end"] / n, overall_acc=(res["start"] + res["end"]) / (2 * n))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--pred-dir", default=os.path.dirname(HERE))
    ap.add_argument("--lf-desc", default=None, help="LineFinder start_end_person_descriptions CSV")
    ap.add_argument("--lf-vis", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()
    P = lambda f: os.path.join(args.pred_dir, f)
    gt = load_gt(args.gt_root)
    methods = {
        "GPT": load_vlm([P("gpt_line_luna_olivia.jsonl"), P("gpt_line_old.jsonl")]),
        "Gemini": load_vlm([P("gemini_line_luna_olivia.jsonl"), P("gemini_line_old.jsonl")]),
        "Kimi": load_vlm([P("kimi_line_luna_olivia.jsonl"), P("kimi_line_old.jsonl")]),
    }
    if args.lf_desc:
        methods["LineFinder"] = load_lf(args.lf_desc, args.lf_vis)
    plans = {m: {"benchmark": ("visibility", "lenient") if m == "LineFinder" else ("presence", "strict"),
                 "unified": ("presence", "strict")} for m in methods}
    judge = Judge(args.cache)
    jobs = []
    for m, pr in methods.items():
        for rule, prompt in plans[m].values():
            score(pr, gt, rule, prompt, judge, collect=jobs)
    judge.run(jobs)
    out = {"n_gt_images": len(gt), "judge_new_calls": judge.calls, "judge_new_errors": judge.errors}
    for m, pr in methods.items():
        out[m] = {name: score(pr, gt, rule, prompt, judge) for name, (rule, prompt) in plans[m].items()}
    json.dump(out, open(args.json_out, "w"), indent=1)
    for m in methods:
        for name in ("benchmark", "unified"):
            r = out[m][name]
            print(f"{m:10s} {name:9s} n={r['n_images']} start={r['start_acc']:.4f} end={r['end_acc']:.4f} "
                  f"overall={r['overall_acc']:.4f} judged={r['judged']} errors={r['judge_errors']} "
                  f"missing_pred={len(r['missing_prediction'])}")


if __name__ == "__main__":
    main()
