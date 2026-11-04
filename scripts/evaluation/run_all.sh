#!/usr/bin/env bash
set -uo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_DIR="${RUN_DIR:-${RESULTS_ROOT:-$HOME/evaluation-results}/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR"

STATUS=0
for script in "$EVAL_DIR"/[0-9][0-9]_*.sh; do
    name="$(basename "$script" .sh)"
    printf '[%s] ===== %s =====\n' "$(date +%H:%M:%S)" "$name"
    if bash "$script"; then
        printf '%s: done\n' "$name" >> "$RUN_DIR/status.txt"
    else
        printf '%s: FAILED (exit %s)\n' "$name" "$?" >> "$RUN_DIR/status.txt"
        STATUS=1
    fi
done

{
    for summary in "$RUN_DIR"/*/summary.txt; do
        printf '\n## %s\n' "$(basename "$(dirname "$summary")")"
        cat "$summary"
    done
    printf '\n## status\n'
    cat "$RUN_DIR/status.txt"
} > "$RUN_DIR/summary.txt"

cat "$RUN_DIR/summary.txt"
printf '\nresults: %s\n' "$RUN_DIR"
exit "$STATUS"
