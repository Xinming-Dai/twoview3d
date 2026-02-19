#!/usr/bin/env python3
"""
Extract a single frame at a specific timestamp from every video under a folder.

Examples:
  python src/twoview3d/preprocess/capture_frames.py --input ./videos --timestamp 12.5 --output ./frames
  python src/twoview3d/preprocess/capture_frames.py -i ./videos -t 00:01:23.200 -o ./frames --mirror
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
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


def parse_timestamp_seconds(ts: str) -> float:
    """
    Accepts:
      - seconds as float/int string: "12", "12.5"
      - timecode: "MM:SS", "HH:MM:SS", each optionally with decimals on seconds
        e.g. "01:23", "00:01:23.200"
    Returns seconds as float.
    """
    ts = ts.strip()
    if re.fullmatch(r"\d+(\.\d+)?", ts):
        return float(ts)

    parts = ts.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Unrecognized timestamp format: {ts!r}")

    try:
        if len(parts) == 2:
            mm = int(parts[0])
            ss = float(parts[1])
            return mm * 60.0 + ss
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
        return hh * 3600.0 + mm * 60.0 + ss
    except ValueError as e:
        raise ValueError(f"Unrecognized timestamp format: {ts!r}") from e


def sanitize_timestamp_for_filename(ts: str) -> str:
    # Keep filenames stable and filesystem-friendly.
    s = ts.strip()
    s = s.replace(":", "-")
    s = s.replace(".", "p")
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", s)
    return s or "t"


def iter_videos(root: Path, exts: set[str]) -> list[Path]:
    vids: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in exts:
            vids.append(p)
    return sorted(vids)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_with_ffmpeg(
    *,
    video_path: Path,
    timestamp: str,
    output_path: Path,
    accurate: bool,
    overwrite: bool,
    square_pixels: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    # -ss before -i is fast seek (less accurate), after -i is accurate seek (slower).
    cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"]
    if not accurate:
        cmd += ["-ss", timestamp]
    cmd += ["-i", str(video_path)]
    if accurate:
        cmd += ["-ss", timestamp]

    ensure_parent(output_path)
    if square_pixels:
        # Bake in sample aspect ratio (SAR) so still images match typical video playback.
        # For SAR=1:1 this is a no-op; for non-square pixels it rescales width by SAR and then sets SAR=1.
        cmd += ["-vf", "scale=iw*sar:ih,setsar=1"]
    cmd += ["-frames:v", "1", "-q:v", "2", str(output_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(msg or f"ffmpeg failed with exit code {proc.returncode}")


def extract_with_opencv(*, video_path: Path, timestamp_seconds: float, output_path: Path, overwrite: bool) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is not installed") from e

    if output_path.exists() and not overwrite:
        return

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError("cannot open video")

        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("cannot read frame at timestamp")

        ensure_parent(output_path)
        ok2 = cv2.imwrite(str(output_path), frame)
        if not ok2:
            raise RuntimeError("failed to write image")
    finally:
        cap.release()


def build_output_path(
    *,
    input_dir: Path,
    output_dir: Path,
    video_path: Path,
    timestamp_tag: str,
    image_ext: str,
    mirror: bool,
) -> Path:
    if mirror:
        rel = video_path.relative_to(input_dir)
        out_base = output_dir / rel.parent
        return out_base / f"{rel.stem}_{timestamp_tag}.{image_ext}"
    # Flat output mode: avoid collisions by prefixing the relative parent path.
    rel = video_path.relative_to(input_dir)
    parent_tag = "_".join(rel.parent.parts)
    if parent_tag:
        return output_dir / f"{parent_tag}__{rel.stem}_{timestamp_tag}.{image_ext}"
    return output_dir / f"{rel.stem}_{timestamp_tag}.{image_ext}"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Extract a frame at a given timestamp from all videos in a folder.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Folder containing videos (searched recursively).")
    p.add_argument(
        "-t",
        "--timestamp",
        required=True,
        type=str,
        help='Timestamp (seconds like "12.5" or timecode like "00:01:23.200").',
    )
    p.add_argument("-o", "--output", required=True, type=Path, help="Output folder for extracted frames.")
    p.add_argument(
        "--backend",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="Extraction backend. auto prefers ffmpeg, falls back to opencv.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use fast seeking (ffmpeg -ss before -i). Less accurate but much faster on long videos.",
    )
    p.add_argument(
        "--square-pixels",
        action="store_true",
        help="(ffmpeg only) Bake in SAR so extracted images use square pixels (matches typical playback).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")
    p.add_argument("--mirror", action="store_true", help="Mirror input subfolders under the output folder.")
    p.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(DEFAULT_VIDEO_EXTS)),
        help='Comma-separated video extensions to include (default: common formats). Example: "mp4,mov,mkv"',
    )
    p.add_argument("--image-ext", type=str, default="jpg", help='Output image extension (e.g. "jpg", "png").')
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files.")
    args = p.parse_args(argv)

    input_dir: Path = args.input
    output_dir: Path = args.output
    timestamp_str: str = args.timestamp.strip()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}", file=sys.stderr)
        return 2

    try:
        timestamp_seconds = parse_timestamp_seconds(timestamp_str)
        if timestamp_seconds < 0:
            raise ValueError("timestamp must be >= 0")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    exts = {e.strip().lower().lstrip(".") for e in str(args.exts).split(",") if e.strip()}
    if not exts:
        print("ERROR: --exts produced an empty set", file=sys.stderr)
        return 2

    videos = iter_videos(input_dir, exts)
    if not videos:
        print(f"No videos found under {input_dir} with extensions: {sorted(exts)}", file=sys.stderr)
        return 1

    ts_tag = f"t{sanitize_timestamp_for_filename(timestamp_str)}"
    image_ext = args.image_ext.strip().lstrip(".") or "jpg"

    ffmpeg_available = shutil.which("ffmpeg") is not None
    use_ffmpeg = args.backend == "ffmpeg" or (args.backend == "auto" and ffmpeg_available)
    use_opencv = args.backend == "opencv" or (args.backend == "auto" and not ffmpeg_available)

    ok_count = 0
    fail_count = 0
    for idx, vp in enumerate(videos, start=1):
        out_path = build_output_path(
            input_dir=input_dir,
            output_dir=output_dir,
            video_path=vp,
            timestamp_tag=ts_tag,
            image_ext=image_ext,
            mirror=bool(args.mirror),
        )

        if args.dry_run:
            print(f"[{idx}/{len(videos)}] {vp} -> {out_path}")
            ok_count += 1
            continue

        try:
            if use_ffmpeg:
                extract_with_ffmpeg(
                    video_path=vp,
                    timestamp=timestamp_str,
                    output_path=out_path,
                    accurate=not bool(args.fast),
                    overwrite=bool(args.overwrite),
                    square_pixels=bool(args.square_pixels),
                )
            elif use_opencv:
                extract_with_opencv(
                    video_path=vp,
                    timestamp_seconds=timestamp_seconds,
                    output_path=out_path,
                    overwrite=bool(args.overwrite),
                )
            else:  # pragma: no cover
                raise RuntimeError("No backend selected")

            print(f"[{idx}/{len(videos)}] OK: {vp} -> {out_path}")
            ok_count += 1
        except Exception as e:
            print(f"[{idx}/{len(videos)}] FAIL: {vp} ({e})", file=sys.stderr)
            fail_count += 1

    print(f"Done. ok={ok_count}, failed={fail_count}")
    return 0 if fail_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
