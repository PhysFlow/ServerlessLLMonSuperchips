#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
setup_run 03_memory
require_models small medium large

MIG="${MIGS[0]}"
PORT="$EVAL_PORT_BASE"
FAILED=0

summary "model | backend | weights on GPU (GB) | KV cache tokens | MIG memory used (GiB) | host memory used (GB)"
for size in small medium large; do
    model_variable="MODEL_${size^^}_PATH"
    model="${!model_variable}"
    for backend in torch hybrid_gemm; do
        name="$size-$backend"
        args=("${COMMON_WORKER_ARGS[@]}")
        [[ "$backend" == "hybrid_gemm" ]] && args+=("${HYBRID_ARGS[@]}")
        host_before="$(host_available_gb)"
        if WORKER_START_MAY_FAIL=1 start_worker "$name" "$MIG" "$PORT" "$model" "${args[@]}"; then
            sleep 5
            log_file="$CURRENT_DIR/$name.log"
            weights="$(sed -n 's/.*Load weight end.*mem usage=\([0-9.]*\) GB.*/\1/p' "$log_file" | head -1)"
            tokens="$(sed -n 's/.*max_total_num_tokens=\([0-9]*\).*/\1/p' "$log_file" | head -1)"
            mig_used="$(mig_memory_used_gib "$MIG")"
            host_used="$(python -c "import sys; print(f'{float(sys.argv[1]) - float(sys.argv[2]):.2f}')" "$host_before" "$(host_available_gb)")"
            stop_worker "$LAST_WORKER_PID"
            summary "$size | $backend | ${weights:-n/a} | ${tokens:-n/a} | $mig_used | $host_used"
        else
            reason="failed to start"
            grep -qi "out of memory\|not enough memory" "$CURRENT_DIR/$name.log" && reason="out of GPU memory"
            summary "$size | $backend | $reason | | |"
            [[ "$backend" == "torch" ]] || FAILED=1
        fi
    done
done
summary "host memory used is the drop in MemAvailable while the worker runs; other workloads on the machine add noise"
exit "$FAILED"
