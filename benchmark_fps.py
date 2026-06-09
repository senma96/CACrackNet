import argparse
import statistics

import torch

from main import get_args_parser, load_checkpoint
from models import build_model


def benchmark(model, batch_size, height, width, warmup, iters):
    x = torch.randn(batch_size, 3, height, width, device="cuda")
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    mean_ms = sum(times) / len(times)
    fps = batch_size * 1000.0 / mean_ms
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95) - 1]
    return mean_ms, p50, p95, fps


def main():
    parser = argparse.ArgumentParser("CACrackNet FPS benchmark", parents=[get_args_parser()])
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--warmup", default=80, type=int)
    parser.add_argument("--iters", default=500, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.device == "cpu":
        raise RuntimeError("FPS benchmark requires CUDA.")

    torch.backends.cudnn.benchmark = True
    args.device = "cuda"
    model, _ = build_model(args)
    model.cuda().eval()
    load_checkpoint(model, args.test_checkpoint, torch.device("cuda"))

    mean_ms, p50, p95, fps = benchmark(
        model,
        args.batch_size,
        args.load_height,
        args.load_width,
        args.warmup,
        args.iters,
    )
    print(f"batch={args.batch_size} input={args.load_height}x{args.load_width}")
    print(f"mean_latency_ms={mean_ms:.4f} p50_ms={p50:.4f} p95_ms={p95:.4f} fps={fps:.2f}")


if __name__ == "__main__":
    main()
