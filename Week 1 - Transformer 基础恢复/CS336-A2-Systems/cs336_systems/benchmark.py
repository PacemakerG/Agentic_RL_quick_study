import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from timeit import default_timer

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


# d_model, d_ff, num_layers, num_heads（PDF 表 1）
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10B": (4608, 12288, 50, 36),
}

parser = argparse.ArgumentParser(description="Transformer FP32 CUDA 端到端测速")
parser.add_argument("--model-size", choices=MODEL_SIZES, default="small")
parser.add_argument("--mode", choices=["forward", "backward", "train"], default="train",
                    help="forward: 仅前向；backward: 前向+反向；train: 前向+反向+AdamW")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--warmup-steps", type=int, default=5)
parser.add_argument("--measure-steps", type=int, default=10)
parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
args = parser.parse_args()
if args.warmup_steps < 0 or min(args.measure_steps, args.batch_size, args.context_length) < 1:
    parser.error("warmup-steps 必须 >= 0，其余数值参数必须 > 0")
if not torch.cuda.is_available():
    parser.error("此测速脚本需要 CUDA GPU")

torch.manual_seed(0)
d_model, d_ff, num_layers, num_heads = MODEL_SIZES[args.model_size]
model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=args.context_length,
    d_model=d_model,
    d_ff=d_ff,
    num_layers=num_layers,
    num_heads=num_heads,
).to(device="cuda", dtype=torch.float32)
model.train(args.mode != "forward")
optimizer = AdamW(model.parameters(), lr=1e-3) if args.mode == "train" else None

# 随机数据只生成一次；标签为右移一个位置的 token。
tokens = torch.randint(0, 10000, (args.batch_size, args.context_length + 1), device="cuda")
x = tokens[:, :-1].contiguous()
targets = tokens[:, 1:].contiguous()

times_ms = []
with torch.set_grad_enabled(args.mode != "forward"):
    for step in range(args.warmup_steps + args.measure_steps):
        # 清理上一步梯度不计时；每步都重新前向，不保留旧计算图。
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = default_timer()
        logits = model(x)
        if args.mode != "forward":
            loss = cross_entropy(logits, targets)
            loss.backward()
        if args.mode == "train":
            optimizer.step()
        torch.cuda.synchronize()
        elapsed_ms = (default_timer() - start) * 1000
        if step >= args.warmup_steps:
            times_ms.append(elapsed_ms)
        del logits
        if args.mode != "forward":
            del loss

# 保存每个测量样本及汇总；重复运行时追加，便于对比预热和模型规模。
row = {
    "gpu": torch.cuda.get_device_name(),
    "torch_version": torch.__version__,
    "model_size": args.model_size,
    "mode": args.mode,
    "batch_size": args.batch_size,
    "context_length": args.context_length,
    "dtype": "float32",
    "warmup_steps": args.warmup_steps,
    "measure_steps": args.measure_steps,
    "mean_ms": mean(times_ms),
    "std_ms": pstdev(times_ms),
    "samples_ms": times_ms,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
write_header = not args.output.exists() or args.output.stat().st_size == 0
with args.output.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=row)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print(f"{args.model_size} {args.mode}: {row['mean_ms']:.3f} ± {row['std_ms']:.3f} ms")
print(f"结果已保存到 {args.output}")
