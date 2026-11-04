#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
setup_run 04_serving
require_models medium

MIG="${MIGS[0]}"
PORT="$EVAL_PORT_BASE"
OUTPUT_LEN=64
INPUT_LENS=(128 2048)
WORKLOADS=("1 8" "4 16" "16 32")
FAILED=0

run_config() {
    local config="$1"
    shift
    if ! WORKER_START_MAY_FAIL=1 start_worker "$config" "$MIG" "$PORT" "$MODEL_MEDIUM_PATH" "${COMMON_WORKER_ARGS[@]}" "$@"; then
        summary "$config | failed to start (see $config.log) | | | |"
        [[ "$config" == "torch" ]] || FAILED=1
        return
    fi
    start_sampling "$config"
    local input_len workload concurrency num_prompts name
    for input_len in "${INPUT_LENS[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            read -r concurrency num_prompts <<< "$workload"
            name="$config-in$input_len-c$concurrency"
            if run_bench "$name" "$PORT" "$input_len" "$OUTPUT_LEN" "$num_prompts" "$concurrency"; then
                summary "$config | $input_len | $concurrency | $(bench_metrics "$name")"
            else
                summary "$config | $input_len | $concurrency | bench failed (see $name.txt) | |"
                FAILED=1
            fi
        done
    done
    stop_sampling
    RATIO_LINES+=("$config: $(ratio_summary "$CURRENT_DIR/$config.log")")
    stop_worker "$LAST_WORKER_PID"
}

RATIO_LINES=()
summary "model: $MODEL_MEDIUM, output $OUTPUT_LEN tokens"
summary "config | input | concurrency | TTFT ms median / p99 | TPOT ms median / p99 | output tokens/s"
run_config hybrid-auto "${HYBRID_ARGS[@]}"
MEMSERVE_AUTO_RATIO=0 run_config hybrid-ratio0 "${HYBRID_ARGS[@]}" --hybrid-gemm-asym-ratio 0
MEMSERVE_AUTO_RATIO=0 run_config hybrid-ratio1 "${HYBRID_ARGS[@]}" --hybrid-gemm-asym-ratio 1
run_config torch
for line in "${RATIO_LINES[@]}"; do
    summary "$line"
done
exit "$FAILED"
