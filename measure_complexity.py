"""
模型复杂度测量脚本：计算 FLOPs、Params 和 Size
Usage:
    python measure_complexity.py
"""

import argparse
import torch
from main import get_args_parser
from models import build_model


def count_parameters(model):
    """统计模型总参数量（不区分是否可训练）"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model, dtype=torch.float32):
    """估算模型以指定精度保存时的文件大小（MB）"""
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.finfo(dtype).bits // 8
    size_mb = total_params * bytes_per_param / (1024 ** 2)
    return size_mb


def measure_flops_thop(model, input_tensor):
    """使用 thop 计算 FLOPs 和 Params"""
    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    except ImportError:
        return None, None


def measure_flops_custom(model, input_tensor):
    """
    自定义 FLOPs 计算（当 thop 不可用时作为 fallback）。
    通过注册 forward hook 统计 Conv2d、Linear、BatchNorm、GroupNorm 等层的计算量。
    """
    flops = 0

    def conv_hook(module, input, output):
        nonlocal flops
        batch_size, in_c, in_h, in_w = input[0].size()
        out_c, out_h, out_w = output[0].size()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (in_c // module.groups)
        output_size = batch_size * out_c * out_h * out_w
        flops += kernel_ops * output_size
        if module.bias is not None:
            flops += output_size

    def linear_hook(module, input, output):
        nonlocal flops
        total_ops = module.in_features * module.out_features
        if module.bias is not None:
            total_ops += module.out_features
        flops += total_ops * input[0].size(0)

    def bn_hook(module, input, output):
        nonlocal flops
        flops += 2 * input[0].numel()

    def gn_hook(module, input, output):
        nonlocal flops
        flops += 2 * input[0].numel()

    hooks = []
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            hooks.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, torch.nn.GroupNorm):
            hooks.append(m.register_forward_hook(gn_hook))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return flops


def format_number(num):
    """将数字格式化为 G / M / K"""
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"


def main():
    parser = argparse.ArgumentParser('Measure Model Complexity', parents=[get_args_parser()])
    args = parser.parse_args()

    args.device = 'cpu'
    args.batch_size_test = 1

    model, _ = build_model(args)
    model.eval()
    model.to(args.device)

    # 构造 dummy input: (1, 3, H, W)
    input_tensor = torch.randn(
        1, 3, args.load_height, args.load_width,
        device=args.device
    )

    # 1. Params
    total_params, trainable_params = count_parameters(model)

    # 2. FLOPs
    # 先尝试用 thop（最准确），否则回退到自定义 hook 统计
    flops_thop, _ = measure_flops_thop(model, input_tensor)
    if flops_thop is not None:
        flops = flops_thop
        print("[INFO] FLOPs calculated via thop.")
    else:
        print("[INFO] thop not installed, using custom FLOPs estimator.")
        print("       (pip install thop for more accurate results)")
        # 重新构建一个干净模型来跑 hook，避免与之前 forward 的副作用
        model_copy, _ = build_model(args)
        model_copy.eval()
        model_copy.to(args.device)
        flops = measure_flops_custom(model_copy, input_tensor)
        del model_copy

    # 3. Size
    size_mb = model_size_mb(model, dtype=torch.float32)

    # 打印结果
    print("\n" + "=" * 60)
    print(f"{'Model Complexity':^60}")
    print("=" * 60)
    print(f"{'FLOPs':<20} {format_number(flops):>30}")
    print(f"{'Params':<20} {format_number(total_params):>30}")
    print(f"{'Size (float32)':<20} {size_mb:.2f}MB")
    print("=" * 60)


if __name__ == '__main__':
    main()
