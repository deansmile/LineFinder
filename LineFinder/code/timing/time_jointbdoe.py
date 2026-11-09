"""Timed JointBDOE pass (person boxes + body orientation in one YOLOv5-based network).

Mirrors the inference in JointBDOE/demos/image.py exactly (same LoadImages, forward
with augment=True, NMS, scale_coords), minus the optional visualization images.
Run from the JointBDOE directory in the `bdoe` environment.

Timed per image: image read + letterbox, forward, NMS, rescale to native coords,
copy to CPU. torch.cuda.synchronize() before and after. Model loading excluded.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # JointBDOE root (utils/, models/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

from common import Timer, TimingLog, image_size, read_list, write_env


def infer_one(model, img_path, imgsz, stride, device, data, args):
    dataset = LoadImages(img_path, img_size=imgsz, stride=stride, auto=True)
    for single_path, img, im0, _ in dataset:
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t / 255.0
        if len(img_t.shape) == 3:
            img_t = img_t[None]
        out_ori = model(img_t, augment=True, scales=args.scales)[0]
        out = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, num_angles=data["num_angles"])
        if out[0] is None or len(out[0]) == 0:
            return None, None
        bboxes = scale_coords(img_t.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()
        orientations = out[0][:, 6:].cpu().numpy() * 360.0
        return bboxes, orientations
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timing-csv", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/JointBDOE_weaklabel_coco.yaml")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf-thres", type=float, default=0.3)
    ap.add_argument("--iou-thres", type=float, default=0.5)
    ap.add_argument("--scales", type=float, nargs="+", default=[1])
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    write_env(args.env_dir, "jointbdoe")
    os.makedirs(args.out_dir, exist_ok=True)
    paths = read_list(args.list)
    with open(args.data) as f:
        data = yaml.safe_load(f)

    device = select_device(args.device, batch_size=1)
    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(args.imgsz, s=stride)
    cuda = device.type == "cuda"

    with torch.no_grad():
        for p in paths[:args.warmup]:
            infer_one(model, p, imgsz, stride, device, data, args)

        log = TimingLog(args.timing_csv)
        for i, p in enumerate(paths):
            image_id = os.path.splitext(os.path.basename(p))[0]
            w, h = image_size(p)
            with Timer(cuda=cuda) as t:
                bboxes, orientations = infer_one(model, p, imgsz, stride, device, data, args)
            log.add("LineFinder", "JointBDOE (YOLOv5 detection + orientation)", image_id, w, h, t.seconds,
                    "separate pass; includes image read+letterbox")
            if bboxes is None:
                print("[INFO] no detections:", image_id)
                continue
            np.save(os.path.join(args.out_dir, image_id + "_bboxes.npy"), bboxes)
            np.save(os.path.join(args.out_dir, image_id + "_orient.npy"), orientations)
            print("[%d/%d] %s %.3fs %d boxes" % (i + 1, len(paths), image_id, t.seconds, len(bboxes)))
        log.close()


if __name__ == "__main__":
    main()
