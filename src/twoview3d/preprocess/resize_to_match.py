#!/usr/bin/env python3
"""
Resize one image to match another image's resolution.

This is useful when you have a high-resolution image and want to downscale it
to exactly the same (width, height) as a reference (usually lower-res) image.

Modes:
  - stretch: resize to exact WxH (may distort aspect ratio)
  - fit:     preserve aspect ratio, pad to target WxH (letterbox)
  - crop:    preserve aspect ratio, center-crop to target WxH (cover)

Examples:
  python src/twoview3d/preprocess/resize_to_match.py --input hi.png --reference lo.png
  python src/twoview3d/preprocess/resize_to_match.py -i hi.jpg -r lo.jpg --mode crop
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _resampling(image_module, *, downscale: bool):
    # Pillow >= 9 uses Image.Resampling, older versions have constants on Image directly.
    Resampling = getattr(image_module, "Resampling", image_module)
    if downscale:
        return getattr(Resampling, "LANCZOS", getattr(image_module, "LANCZOS"))
    return getattr(Resampling, "BICUBIC", getattr(image_module, "BICUBIC"))


def _has_alpha(im) -> bool:
    if im.mode in ("RGBA", "LA"):
        return True
    # Palette images can have transparency.
    return "transparency" in (im.info or {})


def _aspect_ratio_str(w: int, h: int) -> str:
    if h == 0:
        return "undefined"
    g = math.gcd(int(w), int(h)) or 1
    rw = int(w) // g
    rh = int(h) // g
    return f"{(w / h):.6f} ({rw}:{rh})"


def _auto_output_path(
    *,
    input_path: Path,
    reference_path: Path,
    target_w: int,
    target_h: int,
    mode: str,
    overwrite: bool,
) -> Path:
    """
    Create an output filename next to input_path.
    Example: input.jpg -> input__matched_640x480_crop.jpg
    If overwrite=False, adds _v2, _v3, ... if needed.
    """
    in_ext = input_path.suffix.lower()
    ext = in_ext if in_ext else ".png"
    ref_tag = reference_path.stem
    base = f"{input_path.stem}__matched_{target_w}x{target_h}_{mode}__ref-{ref_tag}"
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


def resize_to_match(
    *,
    input_path: Path,
    reference_path: Path,
    output_path: Path | None,
    mode: str,
    allow_upscale: bool,
    apply_exif_orientation: bool,
    background: str,
    jpeg_quality: int,
    overwrite: bool,
) -> dict[str, object]:
    """
    Resizes input image to match reference resolution and writes to output_path.
    If output_path is None, a filename is auto-generated next to the input image.
    Returns a dict containing paths, dimensions, and aspect ratios.
    """
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Pillow is not installed. Install with: pip install "pillow"') from e

    with Image.open(reference_path) as ref:
        if apply_exif_orientation:
            ref = ImageOps.exif_transpose(ref)
        ref_w, ref_h = ref.size
        target_w, target_h = int(ref_w), int(ref_h)

    with Image.open(input_path) as im:
        if apply_exif_orientation:
            im = ImageOps.exif_transpose(im)

        src_w, src_h = im.size
        if (target_w > src_w or target_h > src_h) and not allow_upscale:
            raise ValueError(
                f"reference is larger than input ({target_w}x{target_h} > {src_w}x{src_h}); "
                "refusing to upscale (pass --allow-upscale to override)"
            )

        downscale = target_w < src_w or target_h < src_h
        resample = _resampling(Image, downscale=downscale)

        mode = mode.lower().strip()
        if output_path is None:
            output_path = _auto_output_path(
                input_path=input_path,
                reference_path=reference_path,
                target_w=int(target_w),
                target_h=int(target_h),
                mode=mode,
                overwrite=overwrite,
            )

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {output_path}")

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
            # JPEG cannot store alpha; convert safely.
            if _has_alpha(out):
                bg = Image.new("RGB", out.size, color=background)
                if out.mode != "RGBA":
                    out_rgba = out.convert("RGBA")
                else:
                    out_rgba = out
                bg.paste(out_rgba, mask=out_rgba.split()[-1])
                out_to_save = bg
            else:
                out_to_save = out.convert("RGB") if out.mode != "RGB" else out
            out_to_save.save(output_path, quality=int(jpeg_quality), optimize=True)
        elif ext == "png":
            out.save(output_path, optimize=True)
        else:
            out.save(output_path)

    out_w, out_h = out.size
    return {
        "input_path": str(input_path),
        "reference_path": str(reference_path),
        "output_path": str(output_path),
        "mode": mode,
        "input_size": (int(src_w), int(src_h)),
        "reference_size": (int(ref_w), int(ref_h)),
        "output_size": (int(out_w), int(out_h)),
        "input_aspect_ratio": _aspect_ratio_str(int(src_w), int(src_h)),
        "reference_aspect_ratio": _aspect_ratio_str(int(ref_w), int(ref_h)),
        "output_aspect_ratio": _aspect_ratio_str(int(out_w), int(out_h)),
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Resize an image to match a reference image resolution.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Input image path (the one to resize).")
    p.add_argument("-r", "--reference", required=True, type=Path, help="Reference image path (target resolution).")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Optional output image path. If omitted, an output name is auto-generated next to --input.",
    )
    p.add_argument(
        "--mode",
        choices=("stretch", "fit", "crop"),
        default="stretch",
        help="How to match resolution: stretch (exact), fit (pad), crop (center-crop). Default: stretch.",
    )
    p.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Allow increasing resolution if reference is larger than input (default refuses to upscale).",
    )
    p.add_argument(
        "--apply-exif-orientation",
        action="store_true",
        help="Apply EXIF orientation before resizing (useful for some phone photos).",
    )
    p.add_argument(
        "--background",
        type=str,
        default="white",
        help='Background color for padding / alpha->JPEG conversion (e.g. "white", "black", "#808080"). Default: white.',
    )
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100). Default: 95.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input image not found: {args.input}", file=sys.stderr)
        return 2
    if not args.reference.exists():
        print(f"ERROR: reference image not found: {args.reference}", file=sys.stderr)
        return 2

    try:
        info = resize_to_match(
            input_path=args.input,
            reference_path=args.reference,
            output_path=args.output,
            mode=str(args.mode),
            allow_upscale=bool(args.allow_upscale),
            apply_exif_orientation=bool(args.apply_exif_orientation),
            background=str(args.background),
            jpeg_quality=int(args.jpeg_quality),
            overwrite=bool(args.overwrite),
        )
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    in_w, in_h = info["input_size"]
    ref_w, ref_h = info["reference_size"]
    out_w, out_h = info["output_size"]
    print("Original images")
    print(f'- input:     {info["input_path"]}\t{in_w}x{in_h}')
    print(f'  aspect_ratio: {info["input_aspect_ratio"]}')
    print(f'- reference: {info["reference_path"]}\t{ref_w}x{ref_h}')
    print(f'  aspect_ratio: {info["reference_aspect_ratio"]}')
    print("")
    print("Final image")
    print(f'- output:    {info["output_path"]}\t{out_w}x{out_h}\tmode={info["mode"]}')
    print(f'  aspect_ratio: {info["output_aspect_ratio"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

