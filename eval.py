# eval_videos.py
import argparse
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from pytorch_msssim import ssim as ssim_fn
import lpips
from pytorch_fid import fid_score


# -------------------------
# Video IO (decord -> fallback torchvision)
# -------------------------
def _try_import_decord():
    try:
        import decord  # noqa
        from decord import VideoReader, cpu  # noqa
        return VideoReader, cpu
    except Exception:
        return None, None


def read_video_as_tensor(
    path: Path,
    resize: Optional[int] = None,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> torch.Tensor:
    """
    Returns: float tensor in [0,1], shape [T, 3, H, W]
    """
    VideoReader, cpu = _try_import_decord()
    frames_np = None

    if VideoReader is not None:
        vr = VideoReader(str(path), ctx=cpu(0))
        idx = list(range(0, len(vr), max(1, stride)))
        if max_frames is not None and max_frames > 0:
            idx = idx[:max_frames]
        if len(idx) == 0:
            return torch.empty(0, 3, 0, 0)
        frames_np = vr.get_batch(idx).asnumpy()  # [T,H,W,3] uint8
    else:
        # torchvision fallback
        from torchvision.io import read_video  # requires PyAV/ffmpeg backend

        vframes, _, _ = read_video(str(path), pts_unit="sec")  # [T,H,W,3] uint8
        frames_np = vframes.numpy()
        frames_np = frames_np[:: max(1, stride)]
        if max_frames is not None and max_frames > 0:
            frames_np = frames_np[:max_frames]

    # to torch: [T,3,H,W] float in [0,1]
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous().float() / 255.0

    if resize is not None and resize > 0 and t.numel() > 0:
        t = F.interpolate(t, size=(resize, resize), mode="bilinear", align_corners=False)
    return t


# -------------------------
# Pair discovery
# -------------------------
def find_video_pairs(root: Path, gt_suffix="_gt.mp4", out_suffix="_output.mp4") -> List[Tuple[Path, Path]]:
    gts = sorted(root.glob(f"*{gt_suffix}"))
    pairs = []
    for gt in gts:
        base = gt.name[: -len(gt_suffix)]
        out = root / f"{base}{out_suffix}"
        if out.exists():
            pairs.append((gt, out))
    return pairs


def list_sets(root: Path, gt_suffix="_gt.mp4", out_suffix="_output.mp4") -> Tuple[List[Path], List[Path]]:
    pairs = find_video_pairs(root, gt_suffix, out_suffix)
    gt_paths = [p[0] for p in pairs]
    out_paths = [p[1] for p in pairs]
    return gt_paths, out_paths


# -------------------------
# Frame-wise metrics
# -------------------------
@torch.no_grad()
def compute_psnr_video(gt: torch.Tensor, out: torch.Tensor, eps=1e-10) -> float:
    """
    gt/out: [T,3,H,W] in [0,1]
    """
    T = min(gt.shape[0], out.shape[0])
    if T == 0:
        return float("nan")
    gt = gt[:T]
    out = out[:T]
    mse = torch.mean((gt - out) ** 2, dim=(1, 2, 3)).cpu().numpy()
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, eps))
    return float(np.mean(psnr))


@torch.no_grad()
def compute_ssim_video(gt: torch.Tensor, out: torch.Tensor) -> float:
    """
    Uses pytorch_msssim.ssim frame-wise, averages over frames.
    """
    T = min(gt.shape[0], out.shape[0])
    if T == 0:
        return float("nan")
    gt = gt[:T]
    out = out[:T]

    # compute in chunks to avoid OOM
    chunk = 16
    vals = []
    for i in range(0, T, chunk):
        g = gt[i : i + chunk]
        o = out[i : i + chunk]
        try:
            v = ssim_fn(g, o, data_range=1.0, size_average=False)  # [N]
            vals.append(v.detach().cpu().numpy())
        except TypeError:
            # some versions may not support size_average=False
            v = []
            for j in range(g.shape[0]):
                vj = ssim_fn(g[j : j + 1], o[j : j + 1], data_range=1.0)
                v.append(float(vj.item()))
            vals.append(np.array(v, dtype=np.float32))
    vals = np.concatenate(vals, axis=0)
    return float(np.mean(vals))


@torch.no_grad()
def compute_lpips_video(
    gt: torch.Tensor,
    out: torch.Tensor,
    lpips_model: lpips.LPIPS,
    device: torch.device,
    resize_to: int = 224,
    batch_size: int = 16,
) -> float:
    """
    LPIPS expects [-1,1]. We optionally resize frames to 224 for speed/consistency.
    """
    T = min(gt.shape[0], out.shape[0])
    if T == 0:
        return float("nan")
    gt = gt[:T].to(device)
    out = out[:T].to(device)

    if resize_to is not None and resize_to > 0:
        gt = F.interpolate(gt, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
        out = F.interpolate(out, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

    gt = gt * 2.0 - 1.0
    out = out * 2.0 - 1.0

    vals = []
    for i in range(0, T, batch_size):
        g = gt[i : i + batch_size]
        o = out[i : i + batch_size]
        d = lpips_model(g, o)  # [N,1,1,1] or [N,1]
        vals.append(d.view(-1).detach().cpu().numpy())
    vals = np.concatenate(vals, axis=0)
    return float(np.mean(vals))


# -------------------------
# FID (frame-FID by dumping frames)
# -------------------------
def dump_frames_to_dir(
    video_paths: List[Path],
    out_dir: Path,
    resize: Optional[int],
    stride: int,
    max_total_frames: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for vp in tqdm(video_paths, desc=f"Dumping frames -> {out_dir.name}", leave=False):
        frames = read_video_as_tensor(vp, resize=resize, max_frames=None, stride=stride)  # [T,3,H,W]
        T = frames.shape[0]
        for t in range(T):
            if max_total_frames > 0 and count >= max_total_frames:
                return count
            img = (frames[t].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(img).save(out_dir / f"{count:08d}.png")
            count += 1
    return count


def compute_frame_fid(
    gt_paths: List[Path],
    out_paths: List[Path],
    device: str,
    resize: Optional[int],
    stride: int,
    max_frames: int,
    batch_size: int,
) -> float:
    tmp = Path(tempfile.mkdtemp(prefix="frame_fid_"))
    gt_dir = tmp / "gt"
    out_dir = tmp / "out"
    try:
        dump_frames_to_dir(gt_paths, gt_dir, resize=resize, stride=stride, max_total_frames=max_frames)
        dump_frames_to_dir(out_paths, out_dir, resize=resize, stride=stride, max_total_frames=max_frames)
        fid = fid_score.calculate_fid_given_paths(
            [str(out_dir), str(gt_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )
        return float(fid)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -------------------------
# FVD (cd-fvd)
# -------------------------
def compute_fvd_cdfvd(
    gt_paths: List[Path],
    out_paths: List[Path],
    device: str,
    model: str = "i3d",            # "i3d" or "videomae" :contentReference[oaicite:1]{index=1}
    resolution: int = 256,
    sequence_length: int = 16,
    sample_every_n_frames: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    half_precision: bool = False,
) -> float:
    """
    Uses cd-fvd to compute FVD between two video folders.
    """
    from cdfvd import fvd  # pip install cd-fvd :contentReference[oaicite:2]{index=2}

    tmp = Path(tempfile.mkdtemp(prefix="fvd_"))
    real_dir = tmp / "real"
    fake_dir = tmp / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    try:
        # symlink videos into separate folders
        for i, p in enumerate(gt_paths):
            dst = real_dir / f"{i:05d}{p.suffix}"
            if not dst.exists():
                os.symlink(p.resolve(), dst)
        for i, p in enumerate(out_paths):
            dst = fake_dir / f"{i:05d}{p.suffix}"
            if not dst.exists():
                os.symlink(p.resolve(), dst)

        evaluator = fvd.cdfvd(
            model,
            n_real="full",
            n_fake="full",
            device=device,
            half_precision=half_precision,
        )

        real_loader = evaluator.load_videos(
            str(real_dir),
            resolution=resolution,
            sequence_length=sequence_length,
            sample_every_n_frames=sample_every_n_frames,
            data_type="video_folder",
            num_workers=num_workers,
            batch_size=batch_size,
        )
        fake_loader = evaluator.load_videos(
            str(fake_dir),
            resolution=resolution,
            sequence_length=sequence_length,
            sample_every_n_frames=sample_every_n_frames,
            data_type="video_folder",
            num_workers=num_workers,
            batch_size=batch_size,
        )

        evaluator.compute_real_stats(real_loader)
        evaluator.compute_fake_stats(fake_loader)
        score = evaluator.compute_fvd_from_stats()
        return float(score)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Folder containing *_gt.mp4 and *_output.mp4")
    parser.add_argument("--metric", type=str, nargs="+", required=True,
                        choices=["psnr", "ssim", "lpips", "fid", "fvd"])
    parser.add_argument("--gt_suffix", type=str, default="_gt.mp4")
    parser.add_argument("--out_suffix", type=str, default="_output.mp4")

    # common
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize", type=int, default=0, help="If >0, resize frames/videos to (resize, resize)")

    # frame-wise
    parser.add_argument("--max_frames_per_video", type=int, default=0,
                        help="If >0, read at most this many frames per video for PSNR/SSIM/LPIPS")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for reading videos")
    parser.add_argument("--lpips_batch", type=int, default=16)
    parser.add_argument("--lpips_resize", type=int, default=224)

    # FID
    parser.add_argument("--fid_stride", type=int, default=5, help="Frame stride for dumping frames for FID")
    parser.add_argument("--fid_max_frames", type=int, default=20000, help="Max total frames per set for FID (0=all)")
    parser.add_argument("--fid_batch", type=int, default=50)

    # FVD
    parser.add_argument("--fvd_model", type=str, default="i3d", choices=["i3d", "videomae"])
    parser.add_argument("--fvd_res", type=int, default=256)
    parser.add_argument("--fvd_len", type=int, default=16)
    parser.add_argument("--fvd_stride", type=int, default=1)
    parser.add_argument("--fvd_batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--half", action="store_true")

    args = parser.parse_args()

    resize = args.resize if args.resize > 0 else None
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    pairs = find_video_pairs(args.root, args.gt_suffix, args.out_suffix)
    if len(pairs) == 0:
        raise RuntimeError(f"No pairs found in {args.root} with suffixes {args.gt_suffix} / {args.out_suffix}")

    gt_paths, out_paths = list_sets(args.root, args.gt_suffix, args.out_suffix)

    results: Dict[str, float] = {}

    # Prepare LPIPS model only if needed
    lpips_model = None
    if "lpips" in args.metric:
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    # Frame-wise metrics
    if any(m in args.metric for m in ["psnr", "ssim", "lpips"]):
        psnr_vals, ssim_vals, lpips_vals = [], [], []

        for gt_p, out_p in tqdm(pairs, desc="Frame-wise metrics"):
            gt = read_video_as_tensor(gt_p, resize=resize, max_frames=args.max_frames_per_video, stride=args.stride)
            out = read_video_as_tensor(out_p, resize=resize, max_frames=args.max_frames_per_video, stride=args.stride)
            if gt.numel() == 0 or out.numel() == 0:
                continue

            if "psnr" in args.metric:
                psnr_vals.append(compute_psnr_video(gt, out))
            if "ssim" in args.metric:
                ssim_vals.append(compute_ssim_video(gt, out))
            if "lpips" in args.metric:
                lpips_vals.append(
                    compute_lpips_video(
                        gt, out, lpips_model, device=device,
                        resize_to=args.lpips_resize, batch_size=args.lpips_batch
                    )
                )

        if "psnr" in args.metric:
            results["psnr"] = float(np.nanmean(psnr_vals))
        if "ssim" in args.metric:
            results["ssim"] = float(np.nanmean(ssim_vals))
        if "lpips" in args.metric:
            results["lpips"] = float(np.nanmean(lpips_vals))

    # FID (frame-FID)
    if "fid" in args.metric:
        results["fid"] = compute_frame_fid(
            gt_paths=gt_paths,
            out_paths=out_paths,
            device=str(device),
            resize=resize,
            stride=args.fid_stride,
            max_frames=args.fid_max_frames,
            batch_size=args.fid_batch,
        )

    # FVD
    if "fvd" in args.metric:
        results["fvd"] = compute_fvd_cdfvd(
            gt_paths=gt_paths,
            out_paths=out_paths,
            device=str(device),
            model=args.fvd_model,
            resolution=args.fvd_res,
            sequence_length=args.fvd_len,
            sample_every_n_frames=args.fvd_stride,
            batch_size=args.fvd_batch,
            num_workers=args.num_workers,
            half_precision=args.half,
        )


    save_path = os.path.join(args.root, "results.txt")
    # Print & Save
    print("\n===== Results =====")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("===== Results =====\n")
        for k in args.metric:
            line = f"{k}: {results.get(k, float('nan'))}"
            print(line)
            f.write(line + "\n")

if __name__ == "__main__":
    main()

## Run with the following code
# python eval.py --root /path/to/output/folder --metric psnr ssim lpips fid fvd
