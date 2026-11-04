#!/usr/bin/env bash
STRESS_NAME=soak STRESS_KILL_INTERVAL=0 exec bash "$(dirname "${BASH_SOURCE[0]}")/stress.sh" "${1:-28800}"
