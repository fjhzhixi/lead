import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
from PIL import Image

# Quality presets: (crf, preset)
# high  — near-lossless, largest file
# mid   — good quality, balanced size  (default)
# low   — visibly compressed, smallest file
QUALITY_PRESETS = {
    "high": {"crf": "18", "preset": "slow"},
    "mid":  {"crf": "23", "preset": "medium"},
    "low":  {"crf": "28", "preset": "faster"},
}


def images_to_video(
    image_dir: str,
    output_path: str,
    fps: int = 10,
    strict_continuous: bool = True,
    resize_mismatch: bool = True,
    max_frames: Optional[int] = None,
    exceed_strategy: str = "downsample",
    quality: str = "mid",
    log_fn: Optional[Callable[[str], None]] = print,
) -> str:
    """
    Convert numbered images (e.g., 0000.png, 0001.png, ...) in one folder to a video or GIF.

    Output format is determined by the file extension of output_path:
      - .gif  -> animated GIF via Pillow
      - other -> MP4 video via OpenCV (e.g., .mp4)

    Args:
        image_dir: Directory containing images.
        output_path: Output file path (.mp4, .gif, etc.).
        fps: Frame rate. For GIF, converted to inter-frame duration (ms).
        strict_continuous: If True, require frame indices to be continuous.
        resize_mismatch: If True, resize mismatched frames to the first frame size.
        max_frames: Maximum number of frames to use. If None, use all frames.
        exceed_strategy: How to handle exceeding max_frames.
            - "truncate": Keep only the first max_frames frames.
            - "downsample": Uniformly subsample to max_frames frames.
        quality: Video quality preset — 'high', 'mid', or 'low'.
            high: crf=18, preset=slow  — near-lossless, largest file.
            mid:  crf=23, preset=medium — balanced quality/size (default).
            low:  crf=28, preset=faster — smallest file, visible compression.
        log_fn: Logging function, e.g., print or logger.info. Set to None to disable logs.

    Returns:
        The output file path.

    Raises:
        FileNotFoundError: If image directory does not exist.
        ValueError: If no valid images are found or frame sequence is invalid.
        RuntimeError: If video writer cannot be created.
    """
    def log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality '{quality}'. Choose from: {list(QUALITY_PRESETS)}.")
    q = QUALITY_PRESETS[quality]

    folder = Path(image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    valid_ext = {".png", ".jpg", ".jpeg"}
    numbered: List[Tuple[int, Path]] = []

    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in valid_ext:
            continue
        if not p.stem.isdigit():
            continue
        numbered.append((int(p.stem), p))

    if not numbered:
        raise ValueError("No numbered image files found (expected names like 0000.png).")

    numbered.sort(key=lambda x: x[0])
    indices = [x[0] for x in numbered]

    if strict_continuous:
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1] + 1:
                raise ValueError(
                    f"Frame indices are not continuous: {indices[i - 1]} -> {indices[i]}"
                )

    # Apply frame limit strategy.
    original_count = len(numbered)
    if max_frames is not None and len(numbered) > max_frames:
        if exceed_strategy == "truncate":
            numbered = numbered[:max_frames]
            log(f"Truncated frames from {original_count} to {max_frames}.")
        elif exceed_strategy == "downsample":
            step = original_count // max_frames
            numbered = numbered[::step][:max_frames]
            log(f"Downsampled frames from {original_count} to {len(numbered)} (step={step}).")
        else:
            raise ValueError(
                f"Unknown exceed_strategy: {exceed_strategy}. Choose 'truncate' or 'downsample'."
            )

    first_idx, first_path = numbered[0]
    first_bgr = cv2.imread(str(first_path))
    if first_bgr is None:
        raise ValueError(f"Failed to read first frame: {first_path}")

    height, width = first_bgr.shape[:2]
    is_gif = Path(output_path).suffix.lower() == ".gif"

    log(f"Start encoding {'GIF' if is_gif else 'video'} from {len(numbered)} frames.")
    log(f"First frame index: {first_idx}, size: {width}x{height}, fps: {fps}, quality: {quality}")

    def _read_frame(img_path: Path) -> Optional[object]:
        """Read one frame and resize if needed. Returns None if unreadable."""
        frame = cv2.imread(str(img_path))
        if frame is None:
            log(f"Skip unreadable frame: {img_path.name}")
            return None
        h, w = frame.shape[:2]
        if (w, h) != (width, height):
            if not resize_mismatch:
                raise ValueError(
                    f"Frame size mismatch at {img_path.name}: {w}x{h}, expected {width}x{height}"
                )
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            log(f"Resized frame {img_path.name} from {w}x{h} to {width}x{height}")
        return frame

    if is_gif:
        pil_frames: List[Image.Image] = []
        for _, img_path in numbered:
            bgr = _read_frame(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            raise ValueError("No frames were written to GIF.")

        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
        written = len(pil_frames)
    else:
        # Use ffmpeg for H.264 encoding — universally playable on Ubuntu/Windows/macOS.
        # Write frames as sequentially-named PNGs into a temp dir, then call ffmpeg once.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            written = 0
            for i, (_, img_path) in enumerate(numbered):
                frame = _read_frame(img_path)
                if frame is None:
                    continue
                cv2.imwrite(str(tmp_path / f"{written:06d}.png"), frame)
                written += 1

            if written == 0:
                raise ValueError("No frames were written to video.")

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(tmp_path / "%06d.png"),
                "-c:v", "libx264",
                "-preset", q["preset"],
                "-crf", q["crf"],
                "-pix_fmt", "yuv420p",   # required for broad player compatibility
                "-movflags", "+faststart",  # move moov atom to front for streaming
                output_path,
            ]
            log(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
                )

    log(f"Output saved: {output_path}, written frames: {written}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert numbered images in a folder to an MP4 video or an animated GIF."
    )
    parser.add_argument("image_dir", type=str, help="Input image directory.")
    parser.add_argument(
        "output_path",
        type=str,
        help="Output file path. Use .mp4 for video or .gif for animated GIF.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument(
        "--no-strict-continuous",
        action="store_true",
        help="Allow non-continuous frame indices.",
    )
    parser.add_argument(
        "--no-resize-mismatch",
        action="store_true",
        help="Do not resize frames with mismatched resolution.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable log output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to use. If not set, use all frames.",
    )
    parser.add_argument(
        "--exceed-strategy",
        type=str,
        choices=["truncate", "downsample"],
        default="downsample",
        help="Strategy when frame count exceeds max_frames: 'truncate' keeps first N, 'downsample' uniformly samples N.",
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["high", "mid", "low"],
        default="mid",
        help=(
            "Video quality preset (default: mid). "
            "high: crf=18/slow — near-lossless, largest file. "
            "mid: crf=23/medium — balanced quality/size. "
            "low: crf=28/faster — smallest file, visible compression."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    log_fn = None if args.quiet else print

    try:
        images_to_video(
            image_dir=args.image_dir,
            output_path=args.output_path,
            fps=args.fps,
            strict_continuous=not args.no_strict_continuous,
            resize_mismatch=not args.no_resize_mismatch,
            max_frames=args.max_frames,
            exceed_strategy=args.exceed_strategy,
            quality=args.quality,
            log_fn=log_fn,
        )
        if not args.quiet:
            print("Conversion completed successfully.")
        return 0
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())