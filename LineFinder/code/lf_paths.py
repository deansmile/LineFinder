"""Single place for the LineFinder data root.

Set the LINEFINDER_ROOT environment variable to point scripts at a different copy
of the project (the folder that holds Images/, depth_map/, bbox_orient/, GT_json/,
code/, focal_length_px.txt, ...). It defaults to the Torch working copy.

Scripts in LineFinder/code import it directly; scripts elsewhere add LineFinder/code
to sys.path first.
"""
import os

DEFAULT_LINEFINDER_ROOT = "/scratch/ds5725/linefinder/LineFinder"
LINEFINDER_ROOT = os.environ.get("LINEFINDER_ROOT", DEFAULT_LINEFINDER_ROOT)


def lf_path(*parts):
    """Join path parts onto LINEFINDER_ROOT, e.g. lf_path("depth_map")."""
    return os.path.join(LINEFINDER_ROOT, *parts)
