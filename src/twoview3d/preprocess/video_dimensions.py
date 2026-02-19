#!/usr/bin/env python3
"""
Examine (width, height) of video files under a folder.

Backends:
  - ffprobe (recommended): fast and accurate for metadata
  - opencv: reads via cv2.VideoCapture (fallback)

Examples:
  python src/twoview3d/preprocess/video_dimensions.py --input ./videos
  python src/twoview3d/preprocess/video_dimensions.py -i ./videos --backend ffprobe
  python src/twoview3d/preprocess/video_dimensions.py -i ./videos --summary-only
  python src/twoview3d/preprocess/video_dimensions.py -i ./videos --apply-rotation --csv dims.csv --json dims.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VIDEO_EXTS = {
    "mp4",
    "mov",
    "mkv",
    "avi",
    "webm",
    "m4v",
    "mpg",
    "mpeg",
    "m2ts",
    "mts",
}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int


def iter_videos(root: Path, *, exts: set[str], recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    out: list[Path] = []
    for p in it:
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in exts:
            out.append(p)
    return sorted(out)


def maybe_rel(path: Path, root: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _apply_rotation_if_needed(w: int, h: int, rotate_degrees: int | None) -> tuple[int, int]:
    if rotate_degrees is None:
        return w, h
    r = rotate_degrees % 360
    if r in (90, 270):
        return h, w
    return w, h


def read_video_size_ffprobe(path: Path, *, apply_rotation: bool) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH")

    # Note: rotation is often stored as tag "rotate". This doesn't cover every possible container,
    # but works well for common phone/GoPro footage.
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:stream_tags=rotate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(msg or f"ffprobe failed with exit code {proc.returncode}")

    try:
        payload = json.loads(proc.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError("no video stream found")
        s0 = streams[0] or {}
        w = int(s0.get("width") or 0)
        h = int(s0.get("height") or 0)
        if w <= 0 or h <= 0:
            raise RuntimeError("invalid width/height")
        rot: int | None = None
        if apply_rotation:
            tags = s0.get("tags") or {}
            rot_raw = tags.get("rotate")
            if rot_raw is not None:
                try:
                    rot = int(str(rot_raw).strip())
                except Exception:
                    rot = None
        if apply_rotation:
            w, h = _apply_rotation_if_needed(w, h, rot)
        return w, h
    except Exception as e:
        raise RuntimeError(f"failed to parse ffprobe output: {e}") from e


def read_video_size_opencv(path: Path) -> tuple[int, int]:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is not installed") from e

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("cannot open video")
        w = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
        h = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if w <= 0 or h <= 0:
            raise RuntimeError("invalid width/height")
        return w, h
    finally:
        cap.release()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Print video dimensions (width, height) for files in a folder.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Folder containing videos.")
    p.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(DEFAULT_VIDEO_EXTS)),
        help='Comma-separated video extensions to include (default: common formats). Example: "mp4,mov,mkv"',
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subfolders (default scans recursively).",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "ffprobe", "opencv"),
        default="auto",
        help="How to read dimensions. auto prefers ffprobe, falls back to opencv.",
    )
    p.add_argument(
        "--apply-rotation",
        action="store_true",
        help='If using ffprobe, apply rotation tag ("rotate") to swap w/h for 90/270 degrees.',
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
        help="If >0, limit per-file listing to first N videos (sorted). Default 0 prints all (unless --summary-only).",
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

    video_paths = iter_videos(input_dir, exts=exts, recursive=not bool(args.no_recursive))
    if not video_paths:
        print(f"No videos found under {input_dir} with extensions: {sorted(exts)}", file=sys.stderr)
        return 1

    ffprobe_available = shutil.which("ffprobe") is not None
    use_ffprobe = args.backend == "ffprobe" or (args.backend == "auto" and ffprobe_available)
    use_opencv = args.backend == "opencv" or (args.backend == "auto" and not ffprobe_available)

    max_list = int(args.max_list)
    list_limit = None if max_list <= 0 else max_list

    infos: list[VideoInfo] = []
    fail_count = 0
    size_counts: Counter[tuple[int, int]] = Counter()

    for idx, vp in enumerate(video_paths, start=1):
        try:
            if use_ffprobe:
                w, h = read_video_size_ffprobe(vp, apply_rotation=bool(args.apply_rotation))
            elif use_opencv:
                w, h = read_video_size_opencv(vp)
            else:  # pragma: no cover
                raise RuntimeError("No backend selected")

            infos.append(VideoInfo(path=vp, width=w, height=h))
            size_counts[(w, h)] += 1

            if not args.summary_only and (list_limit is None or idx <= list_limit):
                shown = maybe_rel(vp, input_dir, absolute=bool(args.absolute_paths))
                print(f"[{idx}/{len(video_paths)}] {shown}\t{w}x{h}")
        except Exception as e:
            fail_count += 1
            shown = maybe_rel(vp, input_dir, absolute=bool(args.absolute_paths))
            print(f"[{idx}/{len(video_paths)}] FAIL: {shown} ({e})", file=sys.stderr)

    ok_count = len(infos)
    unique_sizes = len(size_counts)

    print("")
    print("Summary")
    print(f"- folder: {input_dir}")
    print(f"- videos_found: {len(video_paths)}")
    print(f"- ok: {ok_count}")
    print(f"- failed: {fail_count}")
    print(f"- unique_resolutions: {unique_sizes}")
    print(f"- backend: {'ffprobe' if use_ffprobe else 'opencv'}")

    if unique_sizes:
        top = sorted(size_counts.items(), key=lambda kv: (-kv[1], -kv[0][0], -kv[0][1]))
        print("")
        print("Resolution counts (width x height -> count)")
        for (w, h), c in top:
            print(f"- {w}x{h}\t{c}")

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
            "videos_found": len(video_paths),
            "ok": ok_count,
            "failed": fail_count,
            "unique_resolutions": unique_sizes,
            "backend": "ffprobe" if use_ffprobe else "opencv",
            "resolution_counts": [
                {"width": w, "height": h, "count": c}
                for (w, h), c in sorted(size_counts.items(), key=lambda kv: (-kv[1], -kv[0][0], -kv[0][1]))
            ],
            "videos": [
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

