#!/usr/bin/env bash
STRESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STRESS_DIR/../evaluation/common.sh"
setup_run "${STRESS_NAME:-stress}"
require_models small medium large extra
require_free_memory_gb "${STRESS_MIN_FREE_GB:-80}"

DURATION="${1:-7200}"
KILL_INTERVAL="${STRESS_KILL_INTERVAL:-1800}"
RECOVERY="${STRESS_RECOVERY:-900}"
CONCURRENCY="${STRESS_CONCURRENCY:-$((EVAL_MIG_COUNT * 2))}"
MAX_TOKENS="${STRESS_MAX_TOKENS:-16}"
GATEWAY_PORT=$((EVAL_PORT_BASE + 50))
SIZES=(small medium large extra)

MODEL_IDS=()
MODEL_ARGS=()
for index in $(seq 0 "$EVAL_MIG_COUNT"); do
    size="${SIZES[$((index % 4))]}"
    model_id="$size"
    [[ "$index" -lt 4 ]] || model_id="$size-$((index / 4 + 1))"
    path_variable="MODEL_${size^^}_PATH"
    MODEL_IDS+=("$model_id")
    MODEL_ARGS+=("$model_id=${!path_variable}")
done

python - "$CURRENT_DIR/control_plane.json" "$GATEWAY_PORT" "$((EVAL_PORT_BASE + 100))" "${MODEL_ARGS[@]}" <<'EOF'
import json, sys
path, gateway_port, base_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
models = [dict(zip(("model_id", "model_path"), item.split("=", 1))) for item in sys.argv[4:]]
json.dump({
    "gateway": {"host": "127.0.0.1", "port": gateway_port, "queue_timeout": 1800},
    "engine": {"kind": "sglang", "host": "127.0.0.1", "base_port": base_port,
               "startup_timeout": 1800, "request_timeout": 3600},
    "models": models,
}, open(path, "w"), indent=2)
EOF

sample_resources() {
    echo "time,host_available_gb,control_plane_rss_mb,sglang_processes,sglang_rss_mb" > "$CURRENT_DIR/resources.csv"
    while kill -0 "$CONTROL_PLANE_PID" 2>/dev/null; do
        printf '%s,%s,%s,%s,%s\n' "$(date +%s)" "$(host_available_gb)" \
            "$(ps -o rss= -p "$CONTROL_PLANE_PID" | awk '{printf "%.0f", $1 / 1024}')" \
            "$(pgrep -u "$USER" -fc sglang || true)" \
            "$(ps -u "$USER" -o rss=,args= | awk '/sglang/ && !/awk/ {sum += $1} END {printf "%.0f", sum / 1024}')" \
            >> "$CURRENT_DIR/resources.csv"
        sleep 60
    done
}

inject_faults() {
    local pid group
    while sleep "$KILL_INTERVAL"; do
        pid="$(pgrep -P "$CONTROL_PLANE_PID" -f sglang.launch_server | shuf -n 1)" || continue
        [[ -n "$pid" ]] || continue
        group="$(ps -o pgid= -p "$pid" | tr -d ' ')" || continue
        kill -KILL -- "-$group" 2>/dev/null || continue
        printf '%s,%s\n' "$(date +%s)" "$pid" >> "$CURRENT_DIR/faults.csv"
        log "fault injected: killed worker process group $group"
    done
}

summary "models: ${MODEL_ARGS[*]}"
summary "duration $DURATION s, concurrency $CONCURRENCY, max tokens $MAX_TOKENS, kill interval $KILL_INTERVAL s"
start_control_plane "$CURRENT_DIR/control_plane.json" "$GATEWAY_PORT"
start_sampling
sample_resources &
SAMPLER_PIDS+=($!)
if [[ "$KILL_INTERVAL" -gt 0 ]]; then
    inject_faults &
    SAMPLER_PIDS+=($!)
fi

python "$STRESS_DIR/stress_load.py" --url "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
    --duration "$DURATION" --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" \
    --output "$CURRENT_DIR/requests.jsonl" "${MODEL_IDS[@]}"

ALIVE=0
kill -0 "$CONTROL_PLANE_PID" 2>/dev/null && ALIVE=1
stop_sampling
stop_control_plane
python "$STRESS_DIR/stress_report.py" "$CURRENT_DIR" "$ALIVE" "$RECOVERY" > "$CURRENT_DIR/report.txt" || STATUS=$?
while IFS= read -r line; do
    summary "$line"
done < "$CURRENT_DIR/report.txt"
exit "${STATUS:-0}"
