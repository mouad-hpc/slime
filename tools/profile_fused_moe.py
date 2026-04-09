#!/usr/bin/env python3
"""Profile fused MoE backward kernels.

Usage:
  # Print strides + torch profiler summary:
  python3 tools/profile_fused_moe.py

  # With ncu (kernel-level metrics):
  ncu --target-processes all --kernel-name "backward_weight" --set full \
    -o bwd_weight_profile python3 tools/profile_fused_moe.py

  # With nsys (system timeline):
  nsys profile --stats=true -o moe_timeline python3 tools/profile_fused_moe.py
"""
import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/root/slime")
from slime.backends.megatron_utils.kernels.fused_experts import fused_experts_impl


def print_strides(T, H, FFN, E, K):
    grad_output = torch.empty(T * K, H, device="cuda", dtype=torch.bfloat16)
    input_tensor = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
    grad_weight = torch.empty(E, H, FFN, device="cuda", dtype=torch.bfloat16)

    print("=== backward_weight_kernel strides ===")
    print(f"grad_output {list(grad_output.shape)}:")
    print(f"  stride_gom = {grad_output.stride(0)}  (next token)")
    print(f"  stride_gon = {grad_output.stride(1)}  (next hidden dim)")
    print(f"input {list(input_tensor.shape)}:")
    print(f"  stride_im = {input_tensor.stride(0)}  (next token)")
    print(f"  stride_ik = {input_tensor.stride(1)}  (next hidden dim)")
    print(f"grad_weight {list(grad_weight.shape)}:")
    print(f"  stride_gwe = {grad_weight.stride(0)}  (next expert = {grad_weight.stride(0) * 2} bytes)")
    print(f"  stride_gwn = {grad_weight.stride(1)}  (next N)")
    print(f"  stride_gwk = {grad_weight.stride(2)}  (next K)")
    print(f"Atomic contention: {T * K // E} slots/expert, "
          f"{T * K // E // 64} BLOCK_M=64 blocks contending/expert")
    print()

    del grad_output, input_tensor, grad_weight


def make_inputs(T, H, FFN, E, K):
    torch.manual_seed(42)
    hs = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(E, FFN * 2, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(E, H, FFN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tw = torch.softmax(torch.randn(T, K, device="cuda"), -1).to(torch.bfloat16)
    ti = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
    return hs, w1, w2, tw, ti


def warmup(hs, w1, w2, tw, ti, n=3):
    for _ in range(n):
        o = fused_experts_impl(hs, w1, w2, tw, ti)
        o.sum().backward()
        hs.grad = w1.grad = w2.grad = None
    torch.cuda.synchronize()


def run_torch_profiler(hs, w1, w2, tw, ti):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        o = fused_experts_impl(hs, w1, w2, tw, ti)
        o.float().square().mean().backward()
        torch.cuda.synchronize()

    print("=== torch.profiler (sorted by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def run_timed(hs, w1, w2, tw, ti):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    o = fused_experts_impl(hs, w1, w2, tw, ti)
    o.float().square().mean().backward()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"=== fwd+bwd: {ms:.2f} ms, peak_alloc: {peak_gb:.2f} GB ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--no-profiler", action="store_true", help="Skip torch profiler, just time it (for ncu runs)")
    args = parser.parse_args()

    T, H, FFN, E, K = args.num_tokens, args.hidden_size, args.ffn_hidden_size, args.num_experts, args.topk
    print(f"Config: T={T} H={H} FFN={FFN} E={E} K={K}\n")

    print_strides(T, H, FFN, E, K)

    hs, w1, w2, tw, ti = make_inputs(T, H, FFN, E, K)
    warmup(hs, w1, w2, tw, ti)
    hs.grad = w1.grad = w2.grad = None

    if args.no_profiler:
        run_timed(hs, w1, w2, tw, ti)
    else:
        run_torch_profiler(hs, w1, w2, tw, ti)


if __name__ == "__main__":
    main()
