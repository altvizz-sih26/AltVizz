"""
validate_spatial_alignment.py

Validates that a "before" and "during" image pair actually cover the same
real-world area at the same scale, BEFORE they're handed to Person 3's mesh
generation for comparison.

Usage:
    python validate_spatial_alignment.py before.tif during.tif
    python validate_spatial_alignment.py before.tif during.tif --min-overlap 0.95
"""

import argparse
import sys

import rasterio
from rasterio.warp import transform_bounds


def get_image_info(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(path + " has no CRS - not georeferenced, can't validate alignment")
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        res_x, res_y = src.res

        # NEW: if the CRS is geographic (degrees, e.g. EPSG:4326 - common for
        # xBD imagery), res_x/res_y are in degrees, not meters. Left as-is,
        # these tiny values (~0.0000046) round to 0.0m and silently make the
        # resolution check meaningless (always "passes" since 0.0 == 0.0).
        # Convert to approximate meters using the image's own latitude, so
        # the check is actually meaningful regardless of source CRS.
        if src.crs.is_geographic:
            mean_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0
            import math
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))
            res_x_m = abs(res_x) * meters_per_deg_lon
            res_y_m = abs(res_y) * meters_per_deg_lat
        else:
            res_x_m = abs(res_x)
            res_y_m = abs(res_y)

        info = {}
        info["path"] = path
        info["crs"] = str(src.crs)
        info["width"] = src.width
        info["height"] = src.height
        info["bounds_native"] = src.bounds
        info["bounds_wgs84"] = bounds_wgs84
        info["res_x_m"] = res_x_m
        info["res_y_m"] = res_y_m
        return info


def bbox_iou(a, b):
    ax0 = a[0]
    ay0 = a[1]
    ax1 = a[2]
    ay1 = a[3]
    bx0 = b[0]
    by0 = b[1]
    bx1 = b[2]
    by1 = b[3]

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    a_area = (ax1 - ax0) * (ay1 - ay0)
    b_area = (bx1 - bx0) * (by1 - by0)
    union_area = a_area + b_area - inter_area
    if union_area > 0:
        return inter_area / union_area
    return 0.0


def validate_pair(before_path, during_path, min_overlap=0.90, max_res_ratio=1.5):
    before = get_image_info(before_path)
    during = get_image_info(during_path)

    iou = bbox_iou(before["bounds_wgs84"], during["bounds_wgs84"])

    res_before = (before["res_x_m"] + before["res_y_m"]) / 2
    res_during = (during["res_x_m"] + during["res_y_m"]) / 2
    if min(res_before, res_during) > 0:
        res_ratio = max(res_before, res_during) / min(res_before, res_during)
    else:
        res_ratio = float("inf")

    overlap_ok = iou >= min_overlap
    resolution_ok = res_ratio <= max_res_ratio
    passed = overlap_ok and resolution_ok

    report = {}
    report["before"] = before_path
    report["during"] = during_path
    report["before_crs"] = before["crs"]
    report["during_crs"] = during["crs"]
    report["before_bounds_wgs84"] = before["bounds_wgs84"]
    report["during_bounds_wgs84"] = during["bounds_wgs84"]
    report["overlap_iou"] = round(iou, 4)
    report["overlap_ok"] = overlap_ok
    report["min_overlap_required"] = min_overlap
    report["before_resolution_m"] = round(res_before, 3)
    report["during_resolution_m"] = round(res_during, 3)
    report["resolution_ratio"] = round(res_ratio, 2)
    report["resolution_ok"] = resolution_ok
    report["max_resolution_ratio_allowed"] = max_res_ratio
    report["passed"] = passed

    return passed, report


def print_report(report):
    print("")
    print("Before: " + report["before"] + "  (CRS: " + report["before_crs"] + ")")
    print("During: " + report["during"] + "  (CRS: " + report["during_crs"] + ")")
    print("")
    overlap_status = "PASS" if report["overlap_ok"] else "FAIL"
    print("Bounding-box overlap (IoU): " + str(report["overlap_iou"]) + "  " + overlap_status + " (need >= " + str(report["min_overlap_required"]) + ")")
    res_status = "PASS" if report["resolution_ok"] else "FAIL"
    print("Ground resolution: before=" + str(report["before_resolution_m"]) + "m/px, during=" + str(report["during_resolution_m"]) + "m/px, ratio=" + str(report["resolution_ratio"]) + "x  " + res_status + " (need <= " + str(report["max_resolution_ratio_allowed"]) + "x)")
    print("")
    if report["passed"]:
        print("RESULT: PASSED -- safe to send this pair into mesh comparison.")
    else:
        print("RESULT: FAILED -- do NOT send this pair into mesh comparison as-is.")
        if not report["overlap_ok"]:
            print("  -> Images don't cover the same real-world area closely enough.")
        if not report["resolution_ok"]:
            print("  -> Images have meaningfully different ground resolution; resample to match before comparing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("during")
    parser.add_argument("--min-overlap", type=float, default=0.90)
    parser.add_argument("--max-res-ratio", type=float, default=1.5)
    args = parser.parse_args()

    try:
        passed, report = validate_pair(args.before, args.during, args.min_overlap, args.max_res_ratio)
        print_report(report)
        if passed:
            sys.exit(0)
        else:
            sys.exit(1)
    except ValueError as e:
        print("HARD FAIL: " + str(e))
        sys.exit(2)
