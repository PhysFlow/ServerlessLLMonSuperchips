#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
setup_run 01_correctness
require_models small

MIG="${MIGS[0]}"
WORKER_PORT="$EVAL_PORT_BASE"
GATEWAY_PORT=$((EVAL_PORT_BASE + 50))
FAILED=0

run_gsm8k() {
    local name="$1"
    shift
    start_worker "$name" "$MIG" "$WORKER_PORT" "$MODEL_SMALL_PATH" "${COMMON_WORKER_ARGS[@]}" "$@"
    log "GSM8K on $name"
    (cd "$CURRENT_DIR" && python "$SGLANG_DIR/benchmark/gsm8k/bench_sglang.py" \
        --num-questions 1319 --port "$WORKER_PORT" --result-file "$name-result.jsonl") \
        > "$CURRENT_DIR/$name-gsm8k.txt" 2>&1 < /dev/null || die "GSM8K failed on $name"
    stop_worker "$LAST_WORKER_PID"
    GSM8K_ACCURACY="$(sed -n 's/^Accuracy: //p' "$CURRENT_DIR/$name-gsm8k.txt")"
    [[ -n "$GSM8K_ACCURACY" ]] || die "no accuracy in $name-gsm8k.txt"
}

run_gsm8k gsm8k-torch
TORCH_ACCURACY="$GSM8K_ACCURACY"
run_gsm8k gsm8k-hybrid "${HYBRID_ARGS[@]}"
HYBRID_ACCURACY="$GSM8K_ACCURACY"
if python -c "import sys; a, b = map(float, sys.argv[1:]); sys.exit(0 if b >= a - 0.02 else 1)" \
    "$TORCH_ACCURACY" "$HYBRID_ACCURACY"; then
    summary "GSM8K accuracy: PASS (torch $TORCH_ACCURACY, hybrid_gemm $HYBRID_ACCURACY)"
else
    summary "GSM8K accuracy: FAIL (torch $TORCH_ACCURACY, hybrid_gemm $HYBRID_ACCURACY, allowed drop 0.02)"
    FAILED=1
fi
summary "GSM8K hybrid_gemm $(ratio_summary "$CURRENT_DIR/gsm8k-hybrid.log")"

python - "$MODEL_SMALL_PATH" "$GATEWAY_PORT" "$((EVAL_PORT_BASE + 100))" > "$CURRENT_DIR/control_plane.json" <<'EOF'
import json, sys
model_path, gateway_port, base_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
print(json.dumps({
    "gateway": {"host": "127.0.0.1", "port": gateway_port},
    "engine": {"kind": "sglang", "host": "127.0.0.1", "base_port": base_port},
    "models": [{"model_id": "qwen3b", "model_path": model_path}],
}))
EOF

python - "$CURRENT_DIR" <<'EOF'
import json, os, sys
directory = sys.argv[1]
long_text = "MemServe serves many models on MIG instances and keeps their weights in CPU memory. " * 120
payloads = {
    "short": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "long": [{"role": "user", "content": "Summarize the following text in one sentence.\n" + long_text}],
    "multi_turn": [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Got it, your favorite color is blue."},
        {"role": "user", "content": "What is my favorite color?"},
    ],
}
for name, messages in payloads.items():
    with open(os.path.join(directory, f"gateway-{name}-request.json"), "w") as file:
        json.dump({"model": "qwen3b", "messages": messages, "max_tokens": 64}, file)
EOF

start_control_plane "$CURRENT_DIR/control_plane.json" "$GATEWAY_PORT"
for name in short long multi_turn; do
    log "gateway request $name"
    code="$(curl -s -o "$CURRENT_DIR/gateway-$name-response.json" -w '%{http_code}' --max-time 900 \
        -H 'Content-Type: application/json' -d @"$CURRENT_DIR/gateway-$name-request.json" \
        "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions")" || code="000"
    if [[ "$code" == "200" ]] && python -c "import json, sys; content = json.load(open(sys.argv[1]))['choices'][0]['message']['content']; sys.exit(0 if content.strip() else 1)" \
        "$CURRENT_DIR/gateway-$name-response.json" 2>/dev/null; then
        summary "gateway $name: PASS"
    else
        summary "gateway $name: FAIL (HTTP $code)"
        FAILED=1
    fi
done
stop_control_plane
summary "gateway workers $(ratio_summary "$CURRENT_DIR/control_plane.log")"

if pgrep -u "$USER" -f "sglang.launch_server" > /dev/null; then
    summary "control plane shutdown: FAIL (sglang workers are still running)"
    FAILED=1
else
    summary "control plane shutdown: PASS (no leftover workers)"
fi
if grep -q "Traceback" "$CURRENT_DIR/control_plane.log"; then
    summary "control plane log: FAIL (contains a Traceback)"
    FAILED=1
else
    summary "control plane log: PASS"
fi

exit "$FAILED"
