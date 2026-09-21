"""Heuristic removal of two acquisition artifacts found in modality_shortcut_gradcam_task2.py:
black letterbox padding at the image edges, and small burned-in device markers in the corners.
"""

import numpy as np
import cv2


def autocrop_letterbox(gray_u8, black_thresh=8, min_content_frac=0.6):
    """Crop away rows/cols that are uniformly black at the image borders.

    Returns (cropped_array, (top, bottom, left, right)) where the tuple is the
    bounding box kept from the original image. Falls back to the full image if
    the detected content region would be implausibly small (likely a genuinely
    dark radiograph rather than letterbox padding).
    """
    h, w = gray_u8.shape
    row_max = gray_u8.max(axis=1)
    col_max = gray_u8.max(axis=0)

    def find_bounds(profile, thresh):
        nonblack = np.where(profile > thresh)[0]
        if len(nonblack) == 0:
            return 0, len(profile)
        return int(nonblack[0]), int(nonblack[-1]) + 1

    top, bottom = find_bounds(row_max, black_thresh)
    left, right = find_bounds(col_max, black_thresh)

    if (bottom - top) < min_content_frac * h or (right - left) < min_content_frac * w:
        return gray_u8, (0, h, 0, w)

    return gray_u8[top:bottom, left:right], (top, bottom, left, right)


def mask_corner_markers(gray_u8, corner_frac=0.18, min_area=40, max_area=4000, min_fill=0.35):
    """Detect small high-contrast blobs (device stickers/icons) in the four
    corners and flat-fill them with the surrounding corner-patch median.

    This is a shape heuristic (compact, high-contrast, small, near a corner),
    not true marker recognition -- it will occasionally miss markers or, more
    rarely, flatten a small patch of real anatomy. Verify with the paired
    before/after Grad-CAM + pixel-stats diagnostic rather than trusting it blindly.
    """
    h, w = gray_u8.shape
    ch, cw = int(h * corner_frac), int(w * corner_frac)
    out = gray_u8.copy()
    corners = {
        "tl": (0, ch, 0, cw),
        "tr": (0, ch, w - cw, w),
        "bl": (h - ch, h, 0, cw),
        "br": (h - ch, h, w - cw, w),
    }
    for y0, y1, x0, x1 in corners.values():
        patch = gray_u8[y0:y1, x0:x1]
        if patch.size == 0 or patch.std() < 3:
            continue
        _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for candidate in (binary, 255 - binary):
            num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
            for label in range(1, num):
                area = stats[label, cv2.CC_STAT_AREA]
                if not (min_area <= area <= max_area):
                    continue
                bw = stats[label, cv2.CC_STAT_WIDTH]
                bh = stats[label, cv2.CC_STAT_HEIGHT]
                fill_frac = area / float(bw * bh + 1e-6)
                if fill_frac < min_fill:
                    continue
                mask = labels == label
                background = patch[~mask]
                if background.size == 0:
                    continue
                fill_value = float(np.median(background))
                region = out[y0:y1, x0:x1]
                region[mask] = fill_value
                out[y0:y1, x0:x1] = region
    return out


def clean_gray_array(gray_u8, do_crop=True, do_mask=True):
    if do_crop:
        gray_u8, _ = autocrop_letterbox(gray_u8)
    if do_mask:
        gray_u8 = mask_corner_markers(gray_u8)
    return gray_u8
