from lf_paths import LINEFINDER_ROOT  # LINEFINDER_ROOT env var
# line_visibility_location.py
import argparse
import json
import numpy as np
import cv2

from identify_queue_start_end import identify_start_end_bboxes, load_fpx_from_txt


def _bbox_edge_flags(bbox_xyxy, W, H, margin_px):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    near_left   = x1 <= margin_px
    near_right  = x2 >= (W - 1 - margin_px)
    near_top    = y1 <= margin_px
    near_bottom = y2 >= (H - 1 - margin_px)
    touches_any = near_left or near_right or near_top or near_bottom
    return touches_any, near_left, near_right, near_top, near_bottom


def _location_bucket_from_center_x(cx, W):
    # 5 bins: far left | center left | center | center right | far right
    r = cx / max(W, 1)
    if r < 0.2:
        return "far left"
    elif r < 0.4:
        return "center left"
    elif r < 0.6:
        return "center"
    elif r < 0.8:
        return "center right"
    else:
        return "far right"


def endpoint_fields(bbox_xyxy, W, H, margin_px):
    """
    Returns the 3 fields for one endpoint:
      visible? location if visible; direction_to_turn if not visible
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    cx = 0.5 * (x1 + x2)

    touches_any, near_left, near_right, near_top, near_bottom = _bbox_edge_flags(
        bbox_xyxy, W, H, margin_px
    )

    if touches_any:
        visible = "no"
        location = "N/A"

        # Your rule: if close/touch left edge -> turn left else right.
        # (If it touches right edge, that implies end is to the right.)
        if near_left:
            turn = "left"
        else:
            turn = "right"
    else:
        visible = "yes"
        location = _location_bucket_from_center_x(cx, W)
        turn = "N/A"

    return visible, location, turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_id", required=True)
    ap.add_argument("--root", default=LINEFINDER_ROOT)
    ap.add_argument("--margin_px", type=int, default=10,
                    help="Pixels to consider 'close to edge' (default 10)")
    ap.add_argument("--out_json", default=None, help="Optional output json path")
    args = ap.parse_args()

    image_id = args.image_id
    root = args.root

    image_path = f"{root}/Images/ImagesOnline/{image_id}.JPG"
    depth_path = f"{root}/depth_map/{image_id}.npy"
    bbox_path  = f"{root}/bbox_orient/{image_id}_bboxes.npy"
    orient_path = f"{root}/bbox_orient/{image_id}_orient.npy"
    fpx_path = f"{root}/focal_length_px.txt"

    f_px = load_fpx_from_txt(fpx_path, image_id)

    # Get start/end bboxes (start=head, end=tail) from your existing script
    res = identify_start_end_bboxes(
        image_path=image_path,
        depth_npy_path=depth_path,
        bboxes_npy_path=bbox_path,
        orient_npy_path=orient_path,
        f_px=f_px,
    )
    start_bbox = res["start_bbox_xyxy"]  # head
    end_bbox   = res["end_bbox_xyxy"]    # tail

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]

    # END (tail)
    end_visible, end_loc, end_turn = endpoint_fields(end_bbox, W, H, args.margin_px)

    # START (head)
    start_visible, start_loc, start_turn = endpoint_fields(start_bbox, W, H, args.margin_px)

    out = {
        "end_of_line_visible": end_visible,
        "end_of_line_location_if_visible": end_loc,
        "direction_to_turn_to_see_end_if_not_visible": end_turn,

        "start_of_line_visible": start_visible,
        "start_of_line_location_if_visible": start_loc,
        "direction_to_turn_to_see_start_if_not_visible": start_turn,
    }

    print(json.dumps(out, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
