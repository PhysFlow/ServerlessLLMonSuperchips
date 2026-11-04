#!/usr/bin/env bash
# Cold start of one worker per trial, MemServe (weights in pinned CPU memory, HybridGEMM)
# against stock SGLang (weights in HBM), on the first MIG. A trial launches the worker,
# waits for /health, sends one request and stops the worker. Total = ready + first request.
# Weight-load and CUDA graph times come from the SGLang log.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
setup_run 07_cold_start
require_models small medium large

COLD_START_REPEATS="${COLD_START_REPEATS:-3}"
COLD_START_DROP_CACHES="${COLD_START_DROP_CACHES:-1}"
COLD_START_BACKENDS="${COLD_START_BACKENDS:-baseline memserve}"
# Extra SGLang arguments for both backends, e.g. "--cuda-graph-max-bs 32".
read -r -a EXTRA_ARGS <<< "${COLD_START_EXTRA_ARGS:-}"
export WORKER_POLL_SECONDS=0.2 SGLANG_LOG_MS=1
MIG="${MIGS[0]}"
PORT=$((EVAL_PORT_BASE + 70))
FAILED=0

drop_page_cache() {
    [[ "$COLD_START_DROP_CACHES" == "1" ]] || return 0
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null \
        || die "dropping the page cache needs passwordless sudo; set COLD_START_DROP_CACHES=0 to measure warm-cache starts"
}

seconds_since() {
    python -c "import sys, time; print(f'{time.time() - float(sys.argv[1]):.2f}')" "$1"
}

log_phases() {
    python - "$1" <<'EOF'
import re, sys
text = open(sys.argv[1], errors="replace").read()
weights = re.findall(r"Load weight end\. elapsed=([0-9.]+) s", text)
graphs = re.findall(r"Capture cuda graph end\. Time elapsed: ([0-9.]+) s", text)
print(weights[-1] if weights else "nan", graphs[-1] if graphs else "nan")
EOF
}

summary "page cache dropped before each trial: $([[ "$COLD_START_DROP_CACHES" == 1 ]] && echo yes || echo no)"
summary "extra SGLang arguments: ${COLD_START_EXTRA_ARGS:-none}"
summary "backend | model | trial | total s | ready s | weight load s | graph capture s | other init s | first request s | status"
for backend in $COLD_START_BACKENDS; do
    args=("${COMMON_WORKER_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
    [[ "$backend" == "memserve" ]] && args+=("${HYBRID_ARGS[@]}")
    for size in small medium large; do
        path_var="MODEL_${size^^}_PATH"
        model="${!path_var}"
        for trial in $(seq 1 "$COLD_START_REPEATS"); do
            name="$backend-$size-$trial"
            drop_page_cache
            started="$(date +%s.%N)"
            if ! WORKER_START_MAY_FAIL=1 start_worker "$name" "$MIG" "$PORT" "$model" "${args[@]}"; then
                status="$(grep -qi "out of memory" "$CURRENT_DIR/$name.log" && echo OOM || echo failed)"
                summary "$backend | $size | $trial | - | - | - | - | - | - | $status"
                [[ "$status" == OOM ]] || FAILED=1
                break
            fi
            ready="$(seconds_since "$started")"
            read -r code first_request <<< "$(curl -s -o "$CURRENT_DIR/$name.response.json" -w '%{http_code} %{time_total}' \
                --max-time 600 -H 'Content-Type: application/json' \
                -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"max_tokens\": 8}" \
                "http://127.0.0.1:$PORT/v1/chat/completions")" || { code=000; first_request=0; }
            total="$(seconds_since "$started")"
            stop_worker "$LAST_WORKER_PID"
            read -r weights graph <<< "$(log_phases "$CURRENT_DIR/$name.log")"
            other="$(python -c "import sys; r, w, g = map(float, sys.argv[1:]); print(f'{r - w - g:.2f}')" "$ready" "$weights" "$graph")"
            status="$([[ "$code" == 200 ]] && echo ok || echo "HTTP $code")"
            [[ "$code" == 200 ]] || FAILED=1
            summary "$backend | $size | $trial | $total | $ready | $weights | $graph | $other | $first_request | $status"
        done
    done
done

python - "$CURRENT_DIR/summary.txt" <<'EOF' | while IFS= read -r line; do summary "$line"; done
import statistics, sys
rows = [[cell.strip() for cell in line.split("|")] for line in open(sys.argv[1]) if line.count("|") == 9]
groups = {}
for backend, model, _, total, ready, weights, graph, other, first, status in rows[1:]:
    if status == "ok":
        groups.setdefault((backend, model), []).append([float(v) for v in (total, weights, graph, other, first)])
print("backend | model | trials | median total s | weight load s | graph capture s | other init s | first request s")
for (backend, model), values in sorted(groups.items()):
    medians = [statistics.median(column) for column in zip(*values)]
    print(f"{backend} | {model} | {len(values)} | " + " | ".join(f"{value:.1f}" for value in medians))
EOF
exit "$FAILED"
