#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
setup_run 05_startup_switch
require_models small medium large extra
require_free_memory_gb 80

GATEWAY_PORT=$((EVAL_PORT_BASE + 50))
SEQUENCE=(small medium large extra small medium large extra small medium large)
CONTROL_PLANE_TIMEOUT=1500
FAILED=0

python - "$CURRENT_DIR/control_plane.json" "$GATEWAY_PORT" "$((EVAL_PORT_BASE + 100))" "$CONTROL_PLANE_TIMEOUT" \
    "$MODEL_SMALL_PATH" "$MODEL_MEDIUM_PATH" "$MODEL_LARGE_PATH" "$MODEL_EXTRA_PATH" <<'EOF'
import json, sys
path, gateway_port, base_port, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
names = ["small", "medium", "large", "extra"]
json.dump({
    "gateway": {"host": "127.0.0.1", "port": gateway_port, "queue_timeout": timeout},
    "engine": {"kind": "sglang", "host": "127.0.0.1", "base_port": base_port,
               "startup_timeout": timeout, "request_timeout": timeout},
    "models": [{"model_id": name, "model_path": model} for name, model in zip(names, sys.argv[5:])],
}, open(path, "w"))
EOF

start_control_plane "$CURRENT_DIR/control_plane.json" "$GATEWAY_PORT"
start_sampling

summary "step | model | kind | seconds | HTTP"
loaded=0
for index in "${!SEQUENCE[@]}"; do
    model="${SEQUENCE[$index]}"
    if [[ "$loaded" -lt "$EVAL_MIG_COUNT" ]]; then
        kind=cold
    else
        kind=switch
    fi
    loaded=$((loaded + 1))
    log "request $((index + 1)): $model ($kind)"
    result="$(curl -s -o "$CURRENT_DIR/response-$((index + 1))-$model.json" -w '%{http_code} %{time_total}' --max-time 1800 \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"max_tokens\": 8}" \
        "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions")" || result="000 0"
    read -r code seconds <<< "$result"
    summary "$((index + 1)) | $model | $kind | $seconds | $code"
    [[ "$code" == "200" ]] || FAILED=1
done

stop_sampling
stop_control_plane

python - "$CURRENT_DIR/summary.txt" <<'EOF' | while IFS= read -r line; do summary "$line"; done
import statistics, sys
rows = [line.split(" | ") for line in open(sys.argv[1]) if line[:1].isdigit()]
groups = {}
for _, model, kind, seconds, code in rows:
    if code.strip() == "200":
        groups.setdefault((kind, model), []).append(float(seconds))
print("kind | model | count | median seconds")
for (kind, model), values in sorted(groups.items()):
    print(f"{kind} | {model} | {len(values)} | {statistics.median(values):.1f}")
EOF
exit "$FAILED"
