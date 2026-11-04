import collections
import csv
import json
import statistics
import sys
from pathlib import Path


def overlaps_fault(row, faults, recovery):
    return any(row["t"] <= fault + recovery and fault <= row["t"] + row["seconds"] for fault in faults)


def main():
    directory, alive, recovery = Path(sys.argv[1]), sys.argv[2] == "1", float(sys.argv[3])
    print(f"control plane alive at the end: {'yes' if alive else 'NO'}")
    rows = [json.loads(line) for line in open(directory / "requests.jsonl")]
    if not rows:
        print("no request was recorded")
        print("RESULT: FAIL")
        sys.exit(1)
    fault_file = directory / "faults.csv"
    faults = [float(line.split(",")[0]) for line in open(fault_file)] if fault_file.exists() else []
    start = min(row["t"] for row in rows)
    failed = [row for row in rows if row["code"] != 200]
    unexplained = [row for row in failed if not overlaps_fault(row, faults, recovery)]

    print(f"requests {len(rows)}, failed {len(failed)}, faults injected {len(faults)}, "
          f"failed outside the {recovery:.0f} s after a fault: {len(unexplained)}")
    for (model, code), count in sorted(collections.Counter((row["model"], row["code"]) for row in rows).items()):
        print(f"model {model} code {code}: {count}")
    latencies = sorted(row["seconds"] for row in rows if row["code"] == 200 and row["mode"] == "plain")
    if latencies:
        p99 = latencies[max(0, int(len(latencies) * 0.99) - 1)]
        print(f"plain request seconds: p50 {statistics.median(latencies):.1f}, p99 {p99:.1f}")
    windows = collections.Counter(int((row["t"] - start) // 600) * 10 for row in failed)
    print("failures per 10 minutes (start minute: count):", dict(sorted(windows.items())))
    for error, count in collections.Counter(row["error"] for row in unexplained).most_common(5):
        print(f"unexplained failure x{count}: {error}")

    resources = list(csv.DictReader(open(directory / "resources.csv")))
    if len(resources) >= 8:
        quarter = len(resources) // 4
        for column in ("host_available_gb", "control_plane_rss_mb", "sglang_processes", "sglang_rss_mb"):
            values = [float(row[column]) for row in resources if row[column]]
            first = statistics.median(values[:quarter])
            last = statistics.median(values[-quarter:])
            print(f"{column}: first quarter {first:.0f}, last quarter {last:.0f}, change {last - first:+.0f}")

    passed = alive and not unexplained
    print("RESULT:", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
