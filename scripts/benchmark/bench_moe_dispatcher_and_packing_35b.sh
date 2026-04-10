#!/bin/bash
# Benchmark: MoE dispatcher and packing knobs on Qwen3.5-35B-A3B
#
# This script intentionally compares the simple, no-accuracy-loss levers that
# are most likely to matter before writing more Triton:
#   1. `alltoall` vs `flex --moe-enable-deepep`
#   2. static microbatching vs dynamic batching
#   3. log-prob token caps that match the dynamic batch cap
#
# Optional:
#   - If CONTEXT_PARALLEL_SIZE>1 and INCLUDE_ALLGATHER_CP=1, also benchmark the
#     `--allgather-cp` layout.
#
# Usage:
#   bash scripts/benchmark/bench_moe_dispatcher_and_packing_35b.sh
#   NUM_ROLLOUT=20 MAX_TRAINING_STEPS=5 bash scripts/benchmark/bench_moe_dispatcher_and_packing_35b.sh
#
# Compare:
#   grep "actor_train_time" /tmp/bench_moe_*.log
#   grep "log_probs_time" /tmp/bench_moe_*.log
#   grep "train_wait_time" /tmp/bench_moe_*.log

set -euo pipefail
set -x

NUM_ROLLOUT="${NUM_ROLLOUT:-40}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-6}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
DYNAMIC_MAX_TOKENS_PER_GPU="${DYNAMIC_MAX_TOKENS_PER_GPU:-24576}"
DYNAMIC_LOG_PROBS_MAX_TOKENS_PER_GPU="${DYNAMIC_LOG_PROBS_MAX_TOKENS_PER_GPU:-32768}"
INCLUDE_ALLGATHER_CP="${INCLUDE_ALLGATHER_CP:-0}"
PROFILE_TARGET="${PROFILE_TARGET:-}"

cleanup() {
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
}

cleanup

if [ ! -f /root/Qwen3.5-35B-A3B/config.json ]; then
    echo "Downloading Qwen3.5-35B-A3B..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='/root/Qwen3.5-35B-A3B')"
fi

if [ ! -f /root/gsm8k/train.parquet ] || ! python3 -c "import pandas as pd; df = pd.read_parquet('/root/gsm8k/train.parquet'); assert 'messages' in df.columns" 2>/dev/null; then
    echo "Preparing gsm8k (chat format)..."
    rm -f /root/gsm8k/train.parquet /root/gsm8k/test.parquet
    python3 examples/lora/prep_gsm8k.py
fi

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 128
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --micro-batch-size 1
   --qkv-format bshd
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-5
   --clip-grad 1.0
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.4
   --sglang-ep-size 8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 512
   --offload-train
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --max-training-steps "${MAX_TRAINING_STEPS}"
)

PROFILE_ARGS=()
if [ -n "${PROFILE_TARGET}" ]; then
    PROFILE_ARGS+=(--use-pytorch-profiler --profile-step-start 1 --profile-step-end 2 --profile-target "${PROFILE_TARGET}")
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

run_variant() {
    local label="$1"
    local dispatcher="$2"
    local use_deepep="$3"
    local use_dynamic_batch="$4"
    local use_allgather_cp="$5"

    local logfile="/tmp/bench_moe_${label}.log"
    local extra_args=()

    extra_args+=(--moe-token-dispatcher-type "${dispatcher}")

    if [ "${use_deepep}" = "1" ]; then
        extra_args+=(--moe-enable-deepep)
    fi

    if [ "${use_dynamic_batch}" = "1" ]; then
        extra_args+=(
            --use-dynamic-batch-size
            --max-tokens-per-gpu "${DYNAMIC_MAX_TOKENS_PER_GPU}"
            --log-probs-max-tokens-per-gpu "${DYNAMIC_LOG_PROBS_MAX_TOKENS_PER_GPU}"
        )
    fi

    if [ "${use_allgather_cp}" = "1" ]; then
        extra_args+=(--allgather-cp)
    fi

    echo ""
    echo "========================================"
    echo "  Run: ${label}"
    echo "  dispatcher=${dispatcher} deepep=${use_deepep} dynamic_batch=${use_dynamic_batch} allgather_cp=${use_allgather_cp}"
    echo "========================================"

    cleanup
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${GPUS_PER_NODE}" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
        --colocate \
        --calculate-per-token-loss \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}" \
        "${PROFILE_ARGS[@]}" \
        "${extra_args[@]}" \
        2>&1 | tee "${logfile}"

    ray stop --force
    sleep 5
}

run_variant "alltoall_static" "alltoall" "0" "0" "0"
run_variant "flex_deepep_static" "flex" "1" "0" "0"
run_variant "alltoall_dynamic" "alltoall" "0" "1" "0"
run_variant "flex_deepep_dynamic" "flex" "1" "1" "0"

if [ "${INCLUDE_ALLGATHER_CP}" = "1" ] && [ "${CONTEXT_PARALLEL_SIZE}" -gt 1 ]; then
    run_variant "alltoall_dynamic_allgather_cp" "alltoall" "0" "1" "1"
    run_variant "flex_deepep_dynamic_allgather_cp" "flex" "1" "1" "1"
fi

echo ""
echo "========================================"
echo "  Results Summary"
echo "========================================"

for label in alltoall_static flex_deepep_static alltoall_dynamic flex_deepep_dynamic; do
    echo ""
    echo "--- ${label} ---"
    rg -N "actor_train_time|log_probs_time|train_wait_time|sleep_time|wake_up_time|update_weights_time|tflops|peak_gb" "/tmp/bench_moe_${label}.log" || true
done

if [ "${INCLUDE_ALLGATHER_CP}" = "1" ] && [ "${CONTEXT_PARALLEL_SIZE}" -gt 1 ]; then
    for label in alltoall_dynamic_allgather_cp flex_deepep_dynamic_allgather_cp; do
        echo ""
        echo "--- ${label} ---"
        rg -N "actor_train_time|log_probs_time|train_wait_time|sleep_time|wake_up_time|update_weights_time|tflops|peak_gb" "/tmp/bench_moe_${label}.log" || true
    done
fi
