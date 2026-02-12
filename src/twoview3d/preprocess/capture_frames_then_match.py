#!/usr/bin/env python3
"""
Pipeline utility:
  1) extract one frame from two videos (capture_frames.py functionality)
  2) compare the two extracted frames' resolutions (image_dimensions.py functionality)
  3) if different, downscale the larger frame to match the smaller (resize_to_match.py functionality)

This is a convenience wrapper so you can run a single command.

Examples:
  python preprocess/capture_frames_then_match.py -i ./videos -t 12.5 -o ./frames
  python preprocess/capture_frames_then_match.py -i ./videos -t 00:01:23.200 -o ./frames --mode crop

Notes:
  - The input folder must contain exactly 2 videos (recursively). Otherwise this script errors out.
  - The resized image output name is auto-generated next to the larger frame (same folder).
  - By default, extracted frames "keep the video stretch ratio" by baking in SAR (square pixels),
    same as capture_frames.py with --square-pixels.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    # This file lives in <repo>/preprocess/; root is parent of preprocess.
    return Path(__file__).resolve().parents[1]


# Ensure we can import sibling scripts as modules when executed as:
#   python preprocess/capture_frames_then_match.py
_ROOT = _repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _aspect_ratio_str(w: int, h: int) -> str:
    if h == 0:
        return "undefined"
    g = math.gcd(int(w), int(h)) or 1
    rw = int(w) // g
    rh = int(h) // g
    return f"{(w / h):.6f} ({rw}:{rh})"


def _read_image_size(path: Path, *, apply_exif_orientation: bool) -> tuple[int, int]:
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Pillow is not installed. Install with: pip install "pillow"') from e

    with Image.open(path) as im:
        if apply_exif_orientation:
            im = ImageOps.exif_transpose(im)
        w, h = im.size
        return int(w), int(h)


def _has_alpha(im) -> bool:
    if getattr(im, "mode", None) in ("RGBA", "LA"):
        return True
    return "transparency" in (getattr(im, "info", None) or {})


def _resampling(image_module, *, downscale: bool):
    # Pillow >= 9 uses Image.Resampling, older versions have constants on Image directly.
    Resampling = getattr(image_module, "Resampling", image_module)
    if downscale:
        return getattr(Resampling, "LANCZOS", getattr(image_module, "LANCZOS"))
    return getattr(Resampling, "BICUBIC", getattr(image_module, "BICUBIC"))


def _auto_output_path_for_size(
    *,
    input_path: Path,
    target_w: int,
    target_h: int,
    mode: str,
    overwrite: bool,
) -> Path:
    """
    Create an output filename next to input_path.
    Example: frame.jpg -> frame__resized_640x480_crop.jpg
    If overwrite=False, adds _v2, _v3, ... if needed.
    """
    in_ext = input_path.suffix.lower()
    ext = in_ext if in_ext else ".png"
    base = f"{input_path.stem}__resized_{target_w}x{target_h}_{mode}"
    out = input_path.with_name(base + ext)
    if overwrite:
        return out
    if not out.exists():
        return out
    for k in range(2, 10_000):
        cand = input_path.with_name(f"{base}_v{k}{ext}")
        if not cand.exists():
            return cand
    raise RuntimeError("could not find an available output filename")


def _resize_image_to_size(
    *,
    input_path: Path,
    output_path: Path | None,
    target_w: int,
    target_h: int,
    mode: str,
    allow_upscale: bool,
    apply_exif_orientation: bool,
    background: str,
    jpeg_quality: int,
    overwrite: bool,
) -> Path:
    """
    Resize an image to explicit (target_w, target_h) and save.
    If output_path is None, an output name is auto-generated next to input_path.
    """
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Pillow is not installed. Install with: pip install "pillow"') from e

    with Image.open(input_path) as im:
        if apply_exif_orientation:
            im = ImageOps.exif_transpose(im)

        src_w, src_h = im.size
        if (target_w > src_w or target_h > src_h) and not allow_upscale:
            raise ValueError(
                f"target is larger than input ({target_w}x{target_h} > {src_w}x{src_h}); refusing to upscale"
            )

        downscale = target_w < src_w or target_h < src_h
        resample = _resampling(Image, downscale=downscale)

        mode = mode.lower().strip()
        if output_path is None:
            output_path = _auto_output_path_for_size(
                input_path=input_path,
                target_w=int(target_w),
                target_h=int(target_h),
                mode=mode,
                overwrite=overwrite,
            )

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output exists (use --overwrite-resized): {output_path}")

        if mode == "stretch":
            out = im.resize((int(target_w), int(target_h)), resample=resample)
        elif mode == "fit":
            contained = ImageOps.contain(im, (int(target_w), int(target_h)), method=resample)
            if _has_alpha(contained):
                canvas = Image.new("RGBA", (int(target_w), int(target_h)), (0, 0, 0, 0))
                paste_mask = contained.split()[-1] if contained.mode == "RGBA" else None
            else:
                canvas = Image.new("RGB", (int(target_w), int(target_h)), color=background)
                paste_mask = None
            x0 = (int(target_w) - contained.size[0]) // 2
            y0 = (int(target_h) - contained.size[1]) // 2
            canvas.paste(contained, (x0, y0), mask=paste_mask)
            out = canvas
        elif mode == "crop":
            out = ImageOps.fit(im, (int(target_w), int(target_h)), method=resample, centering=(0.5, 0.5))
        else:
            raise ValueError(f"unknown --mode {mode!r} (expected: stretch, fit, crop)")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower().lstrip(".")
        if ext in ("jpg", "jpeg"):
            if _has_alpha(out):
                bg = Image.new("RGB", out.size, color=background)
                out_rgba = out.convert("RGBA") if out.mode != "RGBA" else out
                bg.paste(out_rgba, mask=out_rgba.split()[-1])
                out_to_save = bg
            else:
                out_to_save = out.convert("RGB") if out.mode != "RGB" else out
            out_to_save.save(output_path, quality=int(jpeg_quality), optimize=True)
        elif ext == "png":
            out.save(output_path, optimize=True)
        else:
            out.save(output_path)

    return output_path


def _select_two_videos(
    *,
    input_dir: Path,
    exts: set[str],
) -> tuple[Path, Path]:
    from preprocess.capture_frames import iter_videos  # type: ignore

    vids = iter_videos(input_dir, exts)
    if len(vids) != 2:
        raise ValueError(
            f"Expected exactly 2 videos under {input_dir}, found {len(vids)}. "
            "Please ensure the input folder contains exactly 2 videos."
        )
    return vids[0], vids[1]


def _capture_two_frames(
    *,
    input_dir: Path,
    output_dir: Path,
    video_a: Path,
    video_b: Path,
    timestamp: str,
    backend: str,
    fast: bool,
    square_pixels: bool,
    overwrite: bool,
    mirror: bool,
    image_ext: str,
) -> tuple[Path, Path]:
    from preprocess.capture_frames import (  # type: ignore
        build_output_path,
        extract_with_ffmpeg,
        extract_with_opencv,
        parse_timestamp_seconds,
        sanitize_timestamp_for_filename,
    )

    ffmpeg_available = shutil.which("ffmpeg") is not None
    use_ffmpeg = backend == "ffmpeg" or (backend == "auto" and ffmpeg_available)
    use_opencv = backend == "opencv" or (backend == "auto" and not ffmpeg_available)

    if square_pixels and use_opencv:
        raise RuntimeError("square-pixels is only supported with ffmpeg (install ffmpeg or pass --no-square-pixels)")

    ts_tag = f"t{sanitize_timestamp_for_filename(timestamp)}"
    image_ext = image_ext.strip().lstrip(".") or "jpg"

    out_a = build_output_path(
        input_dir=input_dir,
        output_dir=output_dir,
        video_path=video_a,
        timestamp_tag=ts_tag,
        image_ext=image_ext,
        mirror=mirror,
    )
    out_b = build_output_path(
        input_dir=input_dir,
        output_dir=output_dir,
        video_path=video_b,
        timestamp_tag=ts_tag,
        image_ext=image_ext,
        mirror=mirror,
    )

    if use_ffmpeg:
        extract_with_ffmpeg(
            video_path=video_a,
            timestamp=timestamp,
            output_path=out_a,
            accurate=not bool(fast),
            overwrite=overwrite,
            square_pixels=square_pixels,
        )
        extract_with_ffmpeg(
            video_path=video_b,
            timestamp=timestamp,
            output_path=out_b,
            accurate=not bool(fast),
            overwrite=overwrite,
            square_pixels=square_pixels,
        )
    elif use_opencv:
        ts_seconds = parse_timestamp_seconds(timestamp)
        extract_with_opencv(video_path=video_a, timestamp_seconds=ts_seconds, output_path=out_a, overwrite=overwrite)
        extract_with_opencv(video_path=video_b, timestamp_seconds=ts_seconds, output_path=out_b, overwrite=overwrite)
    else:  # pragma: no cover
        raise RuntimeError("No backend selected")

    return out_a, out_b


def main(argv: list[str]) -> int:
    from preprocess.capture_frames import DEFAULT_VIDEO_EXTS  # type: ignore

    p = argparse.ArgumentParser(
        description="Extract 2 frames, compare resolution, and (if needed) downscale one to match the other."
    )
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
        help="Frame extraction backend. auto prefers ffmpeg, falls back to opencv.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use fast seeking (ffmpeg -ss before -i). Less accurate but much faster on long videos.",
    )
    p.add_argument(
        "--square-pixels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(ffmpeg only) Bake in SAR so extracted images use square pixels (matches typical playback). Default: enabled.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames.")
    p.add_argument("--mirror", action="store_true", help="Mirror input subfolders under the output folder.")
    p.add_argument("--image-ext", type=str, default="jpg", help='Extracted frame extension (e.g. "jpg", "png").')
    p.add_argument(
        "--exts",
        type=str,
        default=",".join(sorted(DEFAULT_VIDEO_EXTS)),
        help='Comma-separated video extensions to include when scanning --input. Example: "mp4,mov"',
    )

    # Resize options (delegated to resize_to_match.py)
    p.add_argument(
        "--mode",
        choices=("stretch", "fit", "crop"),
        default="stretch",
        help="How to match resolution if resizing is needed. Default: stretch.",
    )
    p.add_argument(
        "--apply-exif-orientation",
        action="store_true",
        help="Apply EXIF orientation when reading/resizing images (usually unnecessary for extracted frames).",
    )
    p.add_argument(
        "--background",
        type=str,
        default="white",
        help='Background for padding / alpha->JPEG conversion (e.g. "white", "#808080"). Default: white.',
    )
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100). Default: 95.")
    p.add_argument(
        "--overwrite-resized",
        action="store_true",
        help="Allow overwriting if the auto-named resized output already exists.",
    )

    args = p.parse_args(argv)

    input_dir: Path = args.input
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}", file=sys.stderr)
        return 2

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower().lstrip(".") for e in str(args.exts).split(",") if e.strip()}
    if not exts:
        print("ERROR: --exts produced an empty set", file=sys.stderr)
        return 2

    try:
        va, vb = _select_two_videos(
            input_dir=input_dir,
            exts=exts,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        frame_a, frame_b = _capture_two_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            video_a=va,
            video_b=vb,
            timestamp=str(args.timestamp).strip(),
            backend=str(args.backend),
            fast=bool(args.fast),
            square_pixels=bool(args.square_pixels),
            overwrite=bool(args.overwrite),
            mirror=bool(args.mirror),
            image_ext=str(args.image_ext),
        )
    except Exception as e:
        print(f"ERROR: failed to extract frames: {e}", file=sys.stderr)
        return 1

    try:
        wa, ha = _read_image_size(frame_a, apply_exif_orientation=bool(args.apply_exif_orientation))
        wb, hb = _read_image_size(frame_b, apply_exif_orientation=bool(args.apply_exif_orientation))
    except Exception as e:
        print(f"ERROR: failed to read extracted frames: {e}", file=sys.stderr)
        return 1

    print("Extracted frames")
    print(f"- A: {frame_a}\t{wa}x{ha}")
    print(f"  aspect_ratio: {_aspect_ratio_str(wa, ha)}")
    print(f"- B: {frame_b}\t{wb}x{hb}")
    print(f"  aspect_ratio: {_aspect_ratio_str(wb, hb)}")
    print("")

    if (wa, ha) == (wb, hb):
        print("OK: frames already match resolution; no resize needed.")
        return 0

    # Decide which one to downscale.
    area_a = wa * ha
    area_b = wb * hb
    if area_a == area_b:
        # If equal area but different shape, prefer the one with smaller max-dimension as reference.
        ref_is_a = max(wa, ha) <= max(wb, hb)
    else:
        ref_is_a = area_a < area_b

    ref_frame = frame_a if ref_is_a else frame_b
    ref_w, ref_h = (wa, ha) if ref_is_a else (wb, hb)
    other_frame = frame_b if ref_is_a else frame_a
    other_w, other_h = (wb, hb) if ref_is_a else (wa, ha)

    # If reference is larger in either dimension, resizing the "other" to it would require upscaling.
    # In that case, choose a common target that never upscales: (min(w), min(h)) and resize both if needed.
    needs_upscale = ref_w > other_w or ref_h > other_h
    if needs_upscale:
        target_w = min(wa, wb)
        target_h = min(ha, hb)
        print(
            "Note: neither frame is strictly higher-res in both dimensions; "
            f"resizing to common target {target_w}x{target_h} to avoid upscaling."
        )

        final_a = frame_a
        final_b = frame_b
        try:
            if (wa, ha) != (target_w, target_h):
                final_a = _resize_image_to_size(
                    input_path=frame_a,
                    output_path=None,
                    target_w=int(target_w),
                    target_h=int(target_h),
                    mode=str(args.mode),
                    allow_upscale=False,
                    apply_exif_orientation=bool(args.apply_exif_orientation),
                    background=str(args.background),
                    jpeg_quality=int(args.jpeg_quality),
                    overwrite=bool(args.overwrite_resized),
                )
            if (wb, hb) != (target_w, target_h):
                final_b = _resize_image_to_size(
                    input_path=frame_b,
                    output_path=None,
                    target_w=int(target_w),
                    target_h=int(target_h),
                    mode=str(args.mode),
                    allow_upscale=False,
                    apply_exif_orientation=bool(args.apply_exif_orientation),
                    background=str(args.background),
                    jpeg_quality=int(args.jpeg_quality),
                    overwrite=bool(args.overwrite_resized),
                )
        except Exception as e:
            print(f"ERROR: resize failed in common-target mode: {e}", file=sys.stderr)
            return 1

        # Recompute sizes and print.
        wfa, hfa = _read_image_size(final_a, apply_exif_orientation=bool(args.apply_exif_orientation))
        wfb, hfb = _read_image_size(final_b, apply_exif_orientation=bool(args.apply_exif_orientation))
        print("")
        print("Final frames")
        print(f"- A_final: {final_a}\t{wfa}x{hfa}")
        print(f"  aspect_ratio: {_aspect_ratio_str(wfa, hfa)}")
        print(f"- B_final: {final_b}\t{wfb}x{hfb}")
        print(f"  aspect_ratio: {_aspect_ratio_str(wfb, hfb)}")
        return 0

    # Normal case: downscale the larger one to match the smaller one.
    from preprocess.resize_to_match import resize_to_match  # type: ignore

    try:
        info = resize_to_match(
            input_path=other_frame,
            reference_path=ref_frame,
            output_path=None,  # auto-name next to other_frame
            mode=str(args.mode),
            allow_upscale=False,
            apply_exif_orientation=bool(args.apply_exif_orientation),
            background=str(args.background),
            jpeg_quality=int(args.jpeg_quality),
            overwrite=bool(args.overwrite_resized),
        )
    except Exception as e:
        print(f"ERROR: resize failed: {e}", file=sys.stderr)
        return 1

    out_path = Path(str(info["output_path"]))
    out_w, out_h = _read_image_size(out_path, apply_exif_orientation=bool(args.apply_exif_orientation))

    print("Resize performed")
    print(f"- reference: {ref_frame}\t{ref_w}x{ref_h}")
    print(f"  aspect_ratio: {_aspect_ratio_str(ref_w, ref_h)}")
    print(f"- resized:   {out_path}\t{out_w}x{out_h}\tmode={info['mode']}")
    print(f"  aspect_ratio: {_aspect_ratio_str(out_w, out_h)}")
    print("")
    print("Final frames to use")
    if ref_is_a:
        print(f"- A_final: {frame_a}\t{wa}x{ha}")
        print(f"  aspect_ratio: {_aspect_ratio_str(wa, ha)}")
        print(f"- B_final: {out_path}\t{out_w}x{out_h}")
        print(f"  aspect_ratio: {_aspect_ratio_str(out_w, out_h)}")
    else:
        print(f"- A_final: {out_path}\t{out_w}x{out_h}")
        print(f"  aspect_ratio: {_aspect_ratio_str(out_w, out_h)}")
        print(f"- B_final: {frame_b}\t{wb}x{hb}")
        print(f"  aspect_ratio: {_aspect_ratio_str(wb, hb)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

