#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMSERVE_DIR="${MEMSERVE_DIR:-$(cd "$EVAL_DIR/../.." && pwd)}"
SGLANG_DIR="${SGLANG_DIR:-$(cd "$MEMSERVE_DIR/.." && pwd)/sglang}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/evaluation-results}"
EVAL_GPU="${EVAL_GPU:-1}"
EVAL_MIG_PROFILE="${EVAL_MIG_PROFILE:-2g.20gb}"
EVAL_MIG_COUNT="${EVAL_MIG_COUNT:-3}"
EVAL_PORT_BASE="${EVAL_PORT_BASE:-31000}"
EVAL_MIN_FREE_GB="${EVAL_MIN_FREE_GB:-100}"
WORKER_READY_TIMEOUT="${WORKER_READY_TIMEOUT:-900}"
WORKER_POLL_SECONDS="${WORKER_POLL_SECONDS:-2}"
MODEL_SMALL="${MODEL_SMALL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_MEDIUM="${MODEL_MEDIUM:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_LARGE="${MODEL_LARGE:-Qwen/Qwen2.5-14B-Instruct}"
MODEL_EXTRA="${MODEL_EXTRA:-Qwen/Qwen2.5-1.5B-Instruct}"

COMMON_WORKER_ARGS=(--dtype bfloat16 --chunked-prefill-size 1024 --disable-piecewise-cuda-graph)
HYBRID_ARGS=(--linear-gemm-backend hybrid_gemm)

MIGS=()
WORKER_PIDS=()
SAMPLER_PIDS=()
CONTROL_PLANE_PID=""
LAST_WORKER_PID=""
CURRENT_DIR=""

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

summary() {
    printf '%s\n' "$*" | tee -a "$CURRENT_DIR/summary.txt" >&2
}

discover_migs() {
    nvidia-smi -L | awk -v gpu="$EVAL_GPU" -v profile="$EVAL_MIG_PROFILE" '
        /^GPU [0-9]+:/ { current = $2; sub(":", "", current); next }
        /^ *MIG / && current == gpu {
            if ($2 == profile && match($0, /MIG-[0-9a-f-]+/)) print substr($0, RSTART, RLENGTH)
            else print "OTHER"
        }'
}

resolve_model() {
    local model="$1"
    if [[ -d "$model" ]]; then
        printf '%s\n' "$model"
        return
    fi
    python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], local_files_only=True))" "$model" 2>/dev/null \
        || die "model $model is not downloaded"
}

require_models() {
    local size
    for size in "$@"; do
        case "$size" in
            small) MODEL_SMALL_PATH="$(resolve_model "$MODEL_SMALL")" ;;
            medium) MODEL_MEDIUM_PATH="$(resolve_model "$MODEL_MEDIUM")" ;;
            large) MODEL_LARGE_PATH="$(resolve_model "$MODEL_LARGE")" ;;
            extra) MODEL_EXTRA_PATH="$(resolve_model "$MODEL_EXTRA")" ;;
            *) die "unknown model size $size" ;;
        esac
        local variable="MODEL_${size^^}_PATH"
        log "model $size: ${!variable}"
    done
}

check_environment() {
    [[ -f "$MEMSERVE_DIR/.venv/bin/activate" ]] || die "no virtual environment at $MEMSERVE_DIR/.venv"
    source "$MEMSERVE_DIR/.venv/bin/activate"
    command -v nvidia-smi > /dev/null || die "nvidia-smi not found"
    python -c "import hybrid_gemm, memserve_runtime, memserve_control_plane, sglang" 2>/dev/null \
        || die "hybrid_gemm, memserve_runtime, memserve_control_plane or sglang cannot be imported"
    [[ -f "$SGLANG_DIR/benchmark/gsm8k/bench_sglang.py" ]] || die "sglang checkout not found at $SGLANG_DIR"

    mapfile -t MIGS < <(discover_migs)
    if [[ "${#MIGS[@]}" -ne "$EVAL_MIG_COUNT" ]] || printf '%s\n' "${MIGS[@]}" | grep -q OTHER; then
        die "GPU $EVAL_GPU must have exactly $EVAL_MIG_COUNT MIG $EVAL_MIG_PROFILE instances and nothing else"
    fi

    if pgrep -u "$USER" -f "sglang.launch_server|memserve_control_plane" > /dev/null; then
        pgrep -u "$USER" -af "sglang.launch_server|memserve_control_plane" >&2
        die "leftover sglang or control plane processes are running"
    fi
    local port
    for port in $(seq "$EVAL_PORT_BASE" $((EVAL_PORT_BASE + 199))); do
        if ss -ltn "sport = :$port" | grep -q LISTEN; then
            die "port $port is already in use"
        fi
    done
    local free_gb
    free_gb="$(free -g | awk '/^Mem:/ { print $7 }')"
    [[ "$free_gb" -ge "$EVAL_MIN_FREE_GB" ]] || die "only $free_gb GB of memory is available, need $EVAL_MIN_FREE_GB GB"
}

write_environment() {
    local file="$RUN_DIR/env.txt"
    [[ -f "$file" ]] && return
    {
        echo "date: $(date -Iseconds)"
        echo "memserve: $(git -C "$MEMSERVE_DIR" rev-parse HEAD)"
        echo "sglang: $(git -C "$SGLANG_DIR" rev-parse HEAD)"
        echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
        echo "gpu: $EVAL_GPU, mig profile: $EVAL_MIG_PROFILE x $EVAL_MIG_COUNT"
        printf 'mig: %s\n' "${MIGS[@]}"
        echo "model small: $MODEL_SMALL"
        echo "model medium: $MODEL_MEDIUM"
        echo "model large: $MODEL_LARGE"
        echo "model extra: $MODEL_EXTRA"
        nvidia-smi -L
    } > "$file"
}

setup_run() {
    check_environment
    if [[ -z "${RUN_DIR:-}" ]]; then
        RUN_DIR="$RESULTS_ROOT/$(date +%Y%m%d-%H%M%S)"
    fi
    export RUN_DIR
    mkdir -p "$RUN_DIR"
    write_environment
    CURRENT_DIR="$RUN_DIR/$1"
    mkdir -p "$CURRENT_DIR"
    : > "$CURRENT_DIR/summary.txt"
    trap cleanup EXIT
    trap 'exit 130' INT TERM HUP
    log "results in $CURRENT_DIR"
}

stop_process_group() {
    local pid="$1" signal="${2:-TERM}"
    kill -0 "$pid" 2>/dev/null || return 0
    kill -"$signal" -- -"$pid" 2>/dev/null || kill -"$signal" "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

start_worker() {
    local name="$1" mig="$2" port="$3" model="$4"
    shift 4
    local log_file="$CURRENT_DIR/$name.log"
    log "starting worker $name on $mig port $port"
    CUDA_VISIBLE_DEVICES="$mig" setsid python -m sglang.launch_server \
        --model-path "$model" --host 127.0.0.1 --port "$port" "$@" \
        > "$log_file" 2>&1 < /dev/null &
    LAST_WORKER_PID=$!
    WORKER_PIDS+=("$LAST_WORKER_PID")
    local deadline=$((SECONDS + WORKER_READY_TIMEOUT))
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        if ! kill -0 "$LAST_WORKER_PID" 2>/dev/null; then
            worker_start_failed "$name" "exited during startup" "$log_file"
            return 1
        fi
        if [[ "$SECONDS" -ge "$deadline" ]]; then
            worker_start_failed "$name" "was not ready after $WORKER_READY_TIMEOUT s" "$log_file"
            return 1
        fi
        sleep "$WORKER_POLL_SECONDS"
    done
    log "worker $name ready"
}

worker_start_failed() {
    local name="$1" reason="$2" log_file="$3"
    tail -20 "$log_file" >&2
    stop_worker "$LAST_WORKER_PID"
    [[ "${WORKER_START_MAY_FAIL:-0}" == "1" ]] || die "worker $name $reason"
    log "worker $name $reason"
}

mig_memory_used_gib() {
    python -c "import sys, pynvml as nvml; nvml.nvmlInit(); print(f'{nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByUUID(sys.argv[1])).used / 2**30:.2f}')" "$1"
}

host_available_gb() {
    awk '/^MemAvailable:/ { printf "%.2f\n", $2 / 1024 / 1024 }' /proc/meminfo
}

stop_worker() {
    local pid="$1"
    stop_process_group "$pid" TERM
    local remaining=()
    local item
    for item in "${WORKER_PIDS[@]}"; do
        [[ "$item" == "$pid" ]] || remaining+=("$item")
    done
    WORKER_PIDS=("${remaining[@]}")
}

start_control_plane() {
    local config="$1" port="$2"
    log "starting control plane on port $port"
    setsid python -m memserve_control_plane --config "$config" --port "$port" \
        > "$CURRENT_DIR/control_plane.log" 2>&1 < /dev/null &
    CONTROL_PLANE_PID=$!
    local deadline=$((SECONDS + 120))
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        kill -0 "$CONTROL_PLANE_PID" 2>/dev/null || die "control plane exited during startup"
        [[ "$SECONDS" -lt "$deadline" ]] || die "control plane was not ready after 120 s"
        sleep 1
    done
}

stop_control_plane() {
    [[ -n "$CONTROL_PLANE_PID" ]] || return 0
    stop_process_group "$CONTROL_PLANE_PID" INT
    CONTROL_PLANE_PID=""
}

start_sampling() {
    local prefix="${1:+$1-}"
    python "$EVAL_DIR/sample_gpu.py" --mig-csv "$CURRENT_DIR/${prefix}mig.csv" --link-csv "$CURRENT_DIR/${prefix}link.csv" \
        "${MIGS[@]}" 2> "$CURRENT_DIR/${prefix}sampler.err" &
    SAMPLER_PIDS+=($!)
}

stop_sampling() {
    local pid
    for pid in "${SAMPLER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    SAMPLER_PIDS=()
}

require_free_memory_gb() {
    local free_gb
    free_gb="$(free -g | awk '/^Mem:/ { print $7 }')"
    [[ "$free_gb" -ge "$1" ]] || die "only $free_gb GB of memory is available, need $1 GB"
}

run_bench() {
    local name="$1" port="$2" input_len="$3" output_len="$4" num_prompts="$5" concurrency="$6"
    log "bench $name: input $input_len, output $output_len, $num_prompts prompts, concurrency $concurrency"
    python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$port" \
        --dataset-name random-ids --random-range-ratio 1 \
        --random-input-len "$input_len" --random-output-len "$output_len" \
        --num-prompts "$num_prompts" --max-concurrency "$concurrency" \
        --output-file "$CURRENT_DIR/$name.jsonl" \
        > "$CURRENT_DIR/$name.txt" 2>&1 < /dev/null
}

bench_value() {
    python -c "import json, sys; lines = [line for line in open(sys.argv[1]) if line.strip()]; print(json.loads(lines[-1])[sys.argv[2]])" \
        "$CURRENT_DIR/$1.jsonl" "$2"
}

bench_metrics() {
    printf '%.0f / %.0f | %.0f / %.0f | %.1f' \
        "$(bench_value "$1" median_ttft_ms)" "$(bench_value "$1" p99_ttft_ms)" \
        "$(bench_value "$1" median_tpot_ms)" "$(bench_value "$1" p99_tpot_ms)" \
        "$(bench_value "$1" output_throughput)"
}

ratio_summary() {
    local changes final
    changes="$(grep -c "MemServe asym_ratio .* -> " "$1" || true)"
    final="$(sed -n 's/.*MemServe asym_ratio [0-9.]* -> \([0-9.]*\).*/\1/p' "$1" | tail -1)"
    printf 'ratio changes %s, final %s' "$changes" "${final:-unchanged}"
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    stop_sampling
    stop_control_plane
    local pid
    for pid in "${WORKER_PIDS[@]}"; do
        stop_process_group "$pid" TERM
    done
    local leftover
    leftover="$(pgrep -u "$USER" -f "sglang.launch_server" || true)"
    if [[ -n "$leftover" ]]; then
        log "killing leftover workers: $leftover"
        kill -KILL $leftover 2>/dev/null || true
    fi
    exit "$status"
}
