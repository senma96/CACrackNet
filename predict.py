import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from main import get_args_parser, load_checkpoint
from models import build_model


def preprocess(image_path, size):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(Image.fromarray(image)).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser("CACrackNet prediction", parents=[get_args_parser()])
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--threshold", default=0.5, type=float, help="Binarization threshold for saved masks.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = build_model(args)
    model.to(device).eval()
    load_checkpoint(model, args.test_checkpoint, device)

    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        image_paths = [input_path]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in image_paths:
            x = preprocess(image_path, (args.load_width, args.load_height)).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            mask = (prob > args.threshold).astype(np.uint8) * 255
            cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), mask)

    print(f"Saved {len(image_paths)} masks to {output_dir}")


if __name__ == "__main__":
    main()
