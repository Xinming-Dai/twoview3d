#!/usr/bin/env python3
"""
Examine (width, height) of image files under a folder.

Examples:
  python src/twoview3d/preprocess/image_dimensions.py --input ./images
  python src/twoview3d/preprocess/image_dimensions.py -i ./images --no-recursive --summary-only
  python src/twoview3d/preprocess/image_dimensions.py -i ./images --apply-exif-orientation --csv dims.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_EXTS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "tif",
    "tiff",
    "gif",
}


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int
    height: int


def iter_images(root: Path, *, exts: set[str], recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    out: list[Path] = []
    for p in it:
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in exts:
            out.append(p)
    return sorted(out)


def read_image_size(path: Path, *, apply_exif_orientation: bool) -> tuple[int, int]:
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Pillow is not installed. Install with: pip install "pillow"') from e

    with Image.open(path) as im:
        if apply_exif_orientation:
            im = ImageOps.exif_transpose(im)
        w, h = im.size
        return int(w), int(h)


def maybe_rel(path: Path, root: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Print image dimensions (width, height) for files in a folder.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Folder containing images.")
    p.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(DEFAULT_IMAGE_EXTS)),
        help='Comma-separated image extensions to include (default: common formats). Example: "jpg,png,webp"',
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subfolders (default scans recursively).",
    )
    p.add_argument(
        "--apply-exif-orientation",
        action="store_true",
        help="Apply EXIF orientation before reading size (useful for some phone photos).",
    )
    p.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Print absolute paths instead of paths relative to --input.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the summary (no per-file listing).",
    )
    p.add_argument(
        "--max-list",
        type=int,
        default=0,
        help="If >0, limit per-file listing to first N images (sorted). Default 0 prints all (unless --summary-only).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path (columns: path,width,height).",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path (includes per-file results + summary).",
    )
    args = p.parse_args(argv)

    input_dir: Path = args.input
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}", file=sys.stderr)
        return 2

    exts = {e.strip().lower().lstrip(".") for e in str(args.exts).split(",") if e.strip()}
    if not exts:
        print("ERROR: --exts produced an empty set", file=sys.stderr)
        return 2

    image_paths = iter_images(input_dir, exts=exts, recursive=not bool(args.no_recursive))
    if not image_paths:
        print(f"No images found under {input_dir} with extensions: {sorted(exts)}", file=sys.stderr)
        return 1

    max_list = int(args.max_list)
    list_limit = None if max_list <= 0 else max_list

    infos: list[ImageInfo] = []
    fail_count = 0
    size_counts: Counter[tuple[int, int]] = Counter()

    for idx, ip in enumerate(image_paths, start=1):
        try:
            w, h = read_image_size(ip, apply_exif_orientation=bool(args.apply_exif_orientation))
            infos.append(ImageInfo(path=ip, width=w, height=h))
            size_counts[(w, h)] += 1

            if not args.summary_only and (list_limit is None or idx <= list_limit):
                shown = maybe_rel(ip, input_dir, absolute=bool(args.absolute_paths))
                print(f"[{idx}/{len(image_paths)}] {shown}\t{w}x{h}")
        except Exception as e:
            fail_count += 1
            shown = maybe_rel(ip, input_dir, absolute=bool(args.absolute_paths))
            print(f"[{idx}/{len(image_paths)}] FAIL: {shown} ({e})", file=sys.stderr)

    ok_count = len(infos)
    unique_sizes = len(size_counts)

    # Summary
    print("")
    print("Summary")
    print(f"- folder: {input_dir}")
    print(f"- images_found: {len(image_paths)}")
    print(f"- ok: {ok_count}")
    print(f"- failed: {fail_count}")
    print(f"- unique_resolutions: {unique_sizes}")

    if unique_sizes:
        # Sort by count desc, then width desc, then height desc for stable output.
        top = sorted(size_counts.items(), key=lambda kv: (-kv[1], -kv[0][0], -kv[0][1]))
        print("")
        print("Resolution counts (width x height -> count)")
        for (w, h), c in top:
            print(f"- {w}x{h}\t{c}")

    # Optional outputs
    if args.csv is not None:
        out_csv: Path = args.csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["path", "width", "height"])
            for info in infos:
                wr.writerow([maybe_rel(info.path, input_dir, absolute=bool(args.absolute_paths)), info.width, info.height])
        print(f"\nWrote CSV: {out_csv}")

    if args.json is not None:
        out_json: Path = args.json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "folder": str(input_dir),
            "images_found": len(image_paths),
            "ok": ok_count,
            "failed": fail_count,
            "unique_resolutions": unique_sizes,
            "resolution_counts": [
                {"width": w, "height": h, "count": c}
                for (w, h), c in sorted(size_counts.items(), key=lambda kv: (-kv[1], -kv[0][0], -kv[0][1]))
            ],
            "images": [
                {
                    "path": maybe_rel(info.path, input_dir, absolute=bool(args.absolute_paths)),
                    "width": info.width,
                    "height": info.height,
                }
                for info in infos
            ],
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_json}")

    return 0 if fail_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

