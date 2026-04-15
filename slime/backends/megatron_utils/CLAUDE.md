# Megatron Utils — Backend Internals

## Fused MoE Backward Kernels

### What
Custom Triton backward kernels for MoE expert layers, ported from Miles' FSDP backend into slime's Megatron backend. Replaces standard PyTorch autograd with fused kernels for the expert forward/backward pass.

### Files

| File | Role |
|------|------|
| `kernels/fused_moe_triton_backward_kernels.py` | 3 Triton JIT kernels: `backward_input`, `backward_weight`, `backward_topk_weights` |
| `kernels/fused_experts.py` | 4 `torch.autograd.Function` wrappers: GateUpProj, SiluAndMul, DownProj, MoeSumReduce |
| `kernels/fused_moe_integration.py` | Monkey-patches grouped expert modules via `patch_model_for_fused_moe()` |
| `kernels/__init__.py` | Empty — required for relative imports in the package |
| `bridge_lora_helpers.py` | Wires the fused MoE hook via `register_pre_wrap_hook` (after LoRA hook) |

### Pipeline

```
hidden_states
    |
    v
GateUpProjFunction    -- input @ w1.T (SGLang forward kernel, custom Triton backward)
    |                    backward: backward_input_kernel + backward_weight_kernel
    v
SiluAndMulFunction    -- SiLU(gate) * up (SGLang forward, manual PyTorch backward)
    v
DownProjFunction      -- input @ w2.T * topk_weights (SGLang forward, custom Triton backward)
    |                    backward: backward_input + backward_weight + backward_topk_weights
    v
MoeSumReduceFunction  -- sum over top-k experts (SGLang forward, expand backward)
    v
output_hidden_states
```

### Three Triton Backward Kernels

1. **`backward_input_kernel`** — grad w.r.t. hidden states
   - `grad_input[token] = sum(grad_output[token,expert] @ weight[expert])`
   - Uses `tl.atomic_add` for token accumulation (multiple experts -> same token)

2. **`backward_weight_kernel`** — grad w.r.t. expert weights
   - `grad_weight[expert] = input.T @ grad_output` (per expert)
   - Uses `tl.atomic_add` for expert accumulation (multiple token blocks -> same expert)
   - **Dominant bottleneck: 44% of total CUDA time**

3. **`backward_topk_weights_kernel`** — grad w.r.t. routing weights
   - Recomputes forward output in-kernel (`input @ weight.T`) to avoid storing it
   - Only runs for DownProj (`mul_routed_weight=True`)

### Activation and Guards

- CLI flag: `--use-fused-moe-backward`
- Guards in `slime/utils/arguments.py`: requires Bridge mode, BF16, EP=1, expert-TP=1
- Hook order in `bridge_lora_helpers.py`: LoRA hook first, then fused MoE hook
- LoRA on expert weights: currently raises RuntimeError (see LoRA section below)

### SGLang Forward Kernel Dependency

The forward pass uses SGLang's `invoke_fused_moe_kernel` and `moe_align_block_size`. Version-specific kwargs:
- GateUpProj passes `c_sorted=False, filter_expert=True`
- DownProj passes `a_use_tma=False, b_use_tma=False`

Confirmed working on SGLang 0.5.9 (`osmosisdocker/limes:latest`).

### Block Config

Static, no autotuning: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8`. No `num_warps` or `num_stages` set (Triton defaults: 4 warps, 2 stages). Chunk size: 64K tokens per iteration.

### Memory Characteristics

All three kernels are **memory-bandwidth-bound** at both model dimensions — expert GEMMs are tiny. The `tl.atomic_add` contention dominates, not compute.

| Model | H | FFN | E | topk | Layers | Expert GEMM size |
|-------|---|-----|---|------|--------|-----------------|
| Qwen3.5-35B-A3B | 2048 | 512 | 256 | 8 | 40 | 512x2048 |
| Qwen3.5-122B-A10B | 3072 | 1024 | 256 | 8 | 48 | 1024x3072 |

### Integration Pattern

`patch_model_for_fused_moe()` walks the model tree, finds modules with `linear_fc1` and `linear_fc2` attributes containing `weight<N>` parameters (grouped experts), and replaces their `forward` method. Parameter names are preserved — no renames, no new wrappers holding parameters. This keeps Bridge naming, LoRA injection, HF export, and weight sync contracts intact.

### Key Invariants

- `mlp.experts.linear_fc1` / `mlp.experts.linear_fc2` names must stay visible in `named_parameters()`
- Per-expert `weight<N>` suffixes must remain on the grouped linear modules
- Expert weights are stacked at runtime via `torch.stack` for the fused kernel, not pre-stacked
- The patched forward consumes Megatron's fused expert tensors directly (not HF `ModuleList`)

---

## Profiler Results (H200, Qwen3.5-35B shapes)

Config: T=4096, H=2048, FFN=512, E=64, topk=8. Total fwd+bwd per MoE layer: **9.4ms**.

| Kernel | CUDA time | % | Calls | Bottleneck |
|--------|----------|---|-------|-----------|
| `backward_weight` | 4.17ms | 44% | 2 | `tl.atomic_add` — 8 M-blocks per expert contend |
| `backward_input` | 2.49ms | 26% | 2 | `tl.atomic_add` — multiple experts write same token grad |
| `fused_moe_kernel` (fwd) | 0.76ms | 8% | 2 | SGLang, already optimized |
| `backward_topk_weights` | 0.44ms | 5% | 1 | Recomputes forward, OK |
| SiluAndMul backward | 0.37ms | 4% | 1 | PyTorch elementwise, OK |
| `aten::zero_` (grad init) | 0.40ms | 4% | 13 | Wasteful — buffer reuse eliminates |
| Other (mul, add, copy) | 0.73ms | 8% | — | Framework overhead |

Per training step: ~376ms (35B, 40 layers) or ~451ms (122B, 48 layers) for MoE backward alone.

### Atomic Contention Analysis (backward_weight)

For Qwen3.5-35B GateUpProj backward:
- Weight shape: `(E=256, FFN*2=1024, H=2048)`
- Grid: `ceil(T*topk/64) * ceil(1024/64)` = 512 M-blocks * 16 N-blocks = 8192 blocks
- Tokens per expert: `T*topk / E` = `4096*8 / 256` = 128 = 2 M-blocks per expert
- Each does K/BLOCK_K = 2048/32 = 64 atomic writes per K-loop
- Total atomics: 8192 * 64 = 524,288
- Contention: 2 blocks racing per expert (moderate at E=256; worse at smaller E)

---

## E2E Benchmark Post-Mortem (2026-04-10)

### Result: No measurable speedup at EP=8

Benchmark: FFT (no LoRA), TP=2, EP=8, GBS=128, MBS=1, Qwen3.5-35B-A3B, 8xH200.

| Config | Steps | Mean actor_train_time | Median |
|--------|-------|----------------------|--------|
| Fused (V0 kernel) | 41 | 75.3s | 75.2s |
| No-fused (autograd) | 6 | 74.3s | 74.4s |

**Difference: within noise (< 1.5%)**

### Why: Expert backward is <0.5% of step time

MoE expert backward = 9.4ms/layer × 40 layers = **376ms** per step.
Total `actor_train_time` = **75,000ms**. Expert backward = **0.5%**.

Even a 9x kernel-level speedup saves only ~334ms = 0.45% of step time.

### Root cause: The kernel is for FSDP determinism, not Megatron EP>1 performance

1. Miles' `StandardDispatcher` hardcodes `moe_ep_size = 1` — all experts local to one GPU
2. The kernel exists for **bit-exact deterministic** gradients needed by R3 routing replay
3. With EP=8, each rank handles only 32 local experts (256/8) — expert GEMMs are tiny
4. Megatron's alltoall dispatcher + TEGroupedMLP handles EP>1; the fused kernel is redundant there

### Decision: Do NOT merge fused MoE changes into main

The code is correct and works at any EP size, but adds complexity with zero measurable e2e benefit for the Megatron backend. Keep on `mouad/fused-kernels` branch as reference.

### Training step breakdown (FFT, EP=8, Qwen3.5-35B)

| Component | Time (s) | % of step |
|-----------|----------|-----------|
| train_wait (rollout idle) | 63s | 39% |
| actor_train (fwd+bwd+opt) | 75s | 46% |
| log_probs (fwd-only) | 24s | 15% |
| sleep (offload to CPU) | 20s | — |
| update_weights (sync to SGLang) | 12s | — |
| wake_up (restore from CPU) | 7s | — |

The real bottlenecks are: rollout pipeline scheduling, forward/backward through attention + GDN/Mamba layers, log-probs recomputation, and colocate offload overhead.

---

## MoE Routing And Packing Priorities

### What To Benchmark Before New Triton

For the production-style Qwen3.5-35B-A3B recipe, the next low-risk wins are
outside the current fused backward kernels:

1. **Dispatcher A/B**
   - Compare `--moe-token-dispatcher-type alltoall` against
     `--moe-token-dispatcher-type flex --moe-enable-deepep`
   - Focus on `actor_train_time`, `train_wait_time`, and any DeepEP-heavy trace
     regions rather than isolated expert GEMM timings

2. **Dynamic batching**
   - Test `--use-dynamic-batch-size` with `--max-tokens-per-gpu`
   - Pair it with `--log-probs-max-tokens-per-gpu` so the log-prob path is not
     pinned to the static training limit

3. **CP-specific path**
   - Only benchmark `--allgather-cp` when `context_parallel_size > 1`
   - This is a layout choice, not a universal improvement for CP=1 runs

Reference harness:
- `scripts/benchmark/bench_moe_dispatcher_and_packing_35b.sh`

### If Another Triton Kernel Is Still Needed

Do not spend the next cycle on another expert-weight backward rewrite. The more
plausible kernel targets are:

- input-grad accumulation, which still uses atomics
- routing/top-k preprocessing around `moe_align_block_size`
- routing-weight gradient work
- EP>1 scheduling and communication-adjacent hotspots that do not show up in a
  single-expert microbenchmark

The decision rule is simple: if dispatcher, log-prob, offload, or rollout
waiting still dominate, stay out of `kernels/` for now.

---

## Existing Kernels

| File | What |
|------|------|
| `kernels/fp8_kernel.py` | Blockwise FP8 cast (Triton, E4M3, 128x128 blocks) |
| `kernels/int4_qat/` | INT4 fake quant CUDA kernel |
| `megatron_to_hf/processors/quantizer_fp8.py` | FP8 weight quantization for inference export |
| `megatron_to_hf/processors/quantizer_mxfp8.py` | MXFP8 group quantization via SGLang |
