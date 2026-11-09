"""Timed DepthPro pass (metric depth + focal length).

Mirrors ml-depth-pro/batched_depth.py exactly (create_model_and_transforms with
precision=torch.half, load_rgb, transform, model.infer(image, f_px=f_px)).
Run from the ml-depth-pro directory in the `depth-pro` environment.

Timed per image: image read (load_rgb), transform, inference, copy of the depth map
to CPU. torch.cuda.synchronize() before and after. Model loading excluded.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import depth_pro

from common import Timer, TimingLog, image_size, read_list, write_env


def infer_one(model, transform, path):
    image, _, f_px = depth_pro.load_rgb(path)
    image = transform(image)
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    focal = float(prediction["focallength_px"])
    return depth, focal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out-dir", required=True, help="depth_map output folder")
    ap.add_argument("--focal-txt", required=True)
    ap.add_argument("--timing-csv", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    write_env(args.env_dir, "depthpro")
    os.makedirs(args.out_dir, exist_ok=True)
    paths = read_list(args.list)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")  # CPU only for smoke tests
    model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.half)
    model.eval()

    with torch.no_grad():
        for p in paths[:args.warmup]:
            infer_one(model, transform, p)

        log = TimingLog(args.timing_csv)
        with open(args.focal_txt, "w", encoding="utf-8") as focal_log:
            for i, p in enumerate(paths):
                image_id = os.path.splitext(os.path.basename(p))[0]
                w, h = image_size(p)
                with Timer(cuda=cuda) as t:
                    depth, focal = infer_one(model, transform, p)
                log.add("LineFinder", "DepthPro (metric depth)", image_id, w, h, t.seconds,
                        "separate pass; includes image read+transform")
                np.save(os.path.join(args.out_dir, image_id + ".npy"), depth)
                focal_log.write("%s %s\n" % (image_id, focal))
                focal_log.flush()
                print("[%d/%d] %s %.3fs" % (i + 1, len(paths), image_id, t.seconds))
        log.close()


if __name__ == "__main__":
    main()
