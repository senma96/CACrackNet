import argparse
import datetime
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from tqdm import tqdm

import util.misc as utils
from datasets import create_dataset
from engine import train_one_epoch
from eval.evaluate import eval
from models import build_model
from util.logger import get_logger


def get_args_parser():
    parser = argparse.ArgumentParser("CACrackNet", add_help=False)

    parser.add_argument("--dataset_path", default="./data/DeepCrack", help="Dataset root with train_img/train_lab/test_img/test_lab folders.")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    parser.add_argument("--serial_batches", action="store_true")

    parser.add_argument("--load_width", default=512, type=int)
    parser.add_argument("--load_height", default=512, type=int)
    parser.add_argument("--batch_size_train", default=4, type=int)
    parser.add_argument("--batch_size_test", default=1, type=int)

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--sgd", action="store_true")
    parser.add_argument("--lr_scheduler", default="PolyLR", choices=["PolyLR", "StepLR", "CosLR"])
    parser.add_argument("--lr_drop", default=30, type=int)

    parser.add_argument("--BCELoss_ratio", default=0.83, type=float)
    parser.add_argument("--DiceLoss_ratio", default=0.17, type=float)

    parser.add_argument("--output_dir", default="./checkpoints/weights")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--test_checkpoint", default="./weights/DeepCrack/checkpoint_best.pth")
    parser.add_argument("--test_results_dir", default="./results/test")

    # The dataset loader is intentionally kept compatible with the original code.
    parser.add_argument("--dataset_mode", default="crack")
    return parser


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _build_optimizer(args, model):
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    if args.sgd:
        return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def _build_scheduler(args, optimizer):
    if args.lr_scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    if args.lr_scheduler == "CosLR":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    return PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)


def _normalize_prediction(out):
    out = out.astype(np.float32)
    max_value = float(np.max(out))
    if max_value <= 0:
        return np.zeros_like(out, dtype=np.uint8)
    return np.clip(255.0 * out / max_value, 0, 255).astype(np.uint8)


def save_predictions(model, criterion, args, save_root, logger=None):
    args.phase = "test"
    args.batch_size = args.batch_size_test
    device = torch.device(args.device)
    test_loader = create_dataset(args)
    Path(save_root).mkdir(parents=True, exist_ok=True)

    model.eval()
    pbar = tqdm(total=len(test_loader), desc="Testing")
    with torch.no_grad():
        for data in test_loader:
            x = data["image"].to(device)
            target = data["label"].to(device)
            out = model(x)
            loss = criterion(out, target.float())

            target_np = target[0, 0].detach().cpu().numpy()
            out_np = out[0, 0].detach().cpu().numpy()
            root_name = Path(data["A_paths"][0]).stem

            target_img = (255 * target_np / max(float(np.max(target_np)), 1.0)).astype(np.uint8)
            pred_img = _normalize_prediction(out_np)
            cv2.imwrite(str(Path(save_root) / f"{root_name}_lab.png"), target_img)
            cv2.imwrite(str(Path(save_root) / f"{root_name}_pre.png"), pred_img)

            if logger is not None:
                logger.info("loss -> %s", loss.item())
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
    pbar.close()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return checkpoint


def train(args):
    seed_torch(args.seed)
    dataset_name = Path(args.dataset_path).name
    cur_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    run_name = f"{cur_time}_Dataset->{dataset_name}"
    checkpoint_log_dir = Path("./checkpoints") / run_name
    checkpoint_log_dir.mkdir(parents=True, exist_ok=True)

    log_train = get_logger(checkpoint_log_dir, "train")
    log_test = get_logger(checkpoint_log_dir, "test")
    log_eval = get_logger(checkpoint_log_dir, "eval")

    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)

    args.phase = "train"
    args.batch_size = args.batch_size_train
    train_loader = create_dataset(args)
    print(f"The number of training images = {len(train_loader)}")

    optimizer = _build_optimizer(args, model)
    lr_scheduler = _build_scheduler(args, optimizer)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    max_miou = 0
    best_metrics = {"epoch": 0, "mIoU": 0, "ODS": 0, "OIS": 0, "F1": 0, "Precision": 0, "Recall": 0}

    for epoch in range(args.start_epoch, args.epochs):
        print(f"training epoch start -> {epoch}")
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, epoch, args, log_train)
        log_train.info("epoch %s train_loss -> %s", epoch, train_stats["loss"])
        lr_scheduler.step()

        utils.save_on_master({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }, output_dir / "checkpoint.pth")

        save_root = Path(args.results_dir) / run_name / f"results_{epoch}"
        save_predictions(model, criterion, args, save_root, log_test)
        metrics = eval(log_eval, str(save_root), epoch)
        for key, value in metrics.items():
            print(f"{key} -> {value}")

        if max_miou < metrics["mIoU"]:
            max_miou = metrics["mIoU"]
            best_metrics = metrics
            utils.save_on_master({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }, output_dir / "checkpoint_best.pth")
            log_train.info("update and save best model -> %s", epoch)

        print(f"max_mIoU -> {best_metrics['mIoU']}\nmax Epoch -> {best_metrics['epoch']}")

    print("Process time {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))


def test(args):
    seed_torch(args.seed)
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)
    load_checkpoint(model, args.test_checkpoint, device)
    save_predictions(model, criterion, args, args.test_results_dir)
    print(f"Predictions saved to {args.test_results_dir}")


def main(args):
    if args.phase == "test":
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CACrackNet", parents=[get_args_parser()])
    main(parser.parse_args())
