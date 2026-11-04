#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
setup_run 06_interference
require_models medium large
require_free_memory_gb 110

INPUT_LEN=128
OUTPUT_LEN=64
CONCURRENCY=4
NUM_PROMPTS=16

measure_model() {
    local size="$1"
    local model_variable="MODEL_${size^^}_PATH"
    local model="${!model_variable}"
    local workers=() index port

    for index in "${!MIGS[@]}"; do
        port=$((EVAL_PORT_BASE + index))
        MEMSERVE_AUTO_RATIO=0 start_worker "$size-mig$index" "${MIGS[$index]}" "$port" "$model" \
            "${COMMON_WORKER_ARGS[@]}" "${HYBRID_ARGS[@]}" --hybrid-gemm-asym-ratio 1
        workers+=("$LAST_WORKER_PID")
    done
    for index in "${!MIGS[@]}"; do
        run_bench "$size-warmup-mig$index" "$((EVAL_PORT_BASE + index))" "$INPUT_LEN" 16 4 4 \
            || die "warmup failed on $size-mig$index"
    done

    start_sampling "$size-solo"
    for index in "${!MIGS[@]}"; do
        run_bench "$size-solo-mig$index" "$((EVAL_PORT_BASE + index))" \
            "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_PROMPTS" "$CONCURRENCY" || die "solo bench failed on $size-mig$index"
    done
    stop_sampling

    start_sampling "$size-concurrent"
    local benches=()
    for index in "${!MIGS[@]}"; do
        run_bench "$size-concurrent-mig$index" "$((EVAL_PORT_BASE + index))" \
            "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_PROMPTS" "$CONCURRENCY" &
        benches+=($!)
    done
    for index in "${!benches[@]}"; do
        wait "${benches[$index]}" || die "concurrent bench failed on $size-mig$index"
    done
    stop_sampling

    local solo_total=0 concurrent_total=0 solo concurrent
    for index in "${!MIGS[@]}"; do
        solo="$(bench_value "$size-solo-mig$index" output_throughput)"
        concurrent="$(bench_value "$size-concurrent-mig$index" output_throughput)"
        solo_total="$(python -c "print($solo_total + $solo)")"
        concurrent_total="$(python -c "print($concurrent_total + $concurrent)")"
        summary "$size | mig$index | $(printf '%.1f' "$solo") | $(printf '%.1f' "$concurrent") | $(printf '%.0f' "$(bench_value "$size-solo-mig$index" median_tpot_ms)") | $(printf '%.0f' "$(bench_value "$size-concurrent-mig$index" median_tpot_ms)")"
    done
    summary "$size | total | $(printf '%.1f' "$solo_total") | $(printf '%.1f' "$concurrent_total") | interference gap $(python -c "print(f'{(1 - $concurrent_total / $solo_total) * 100:.0f}%')") |"

    for index in "${!workers[@]}"; do
        stop_worker "${workers[$index]}"
    done
}

summary "input $INPUT_LEN, output $OUTPUT_LEN tokens, $NUM_PROMPTS prompts, concurrency $CONCURRENCY per MIG, hybrid_gemm ratio fixed at 1"
summary "model | MIG | solo tokens/s | concurrent tokens/s | solo TPOT ms | concurrent TPOT ms"
measure_model medium
measure_model large
