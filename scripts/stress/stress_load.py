import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

INPUT_WORDS = (1, 50, 300, 1000, 3000)
MODES = ("plain", "stream", "abort")
MODE_WEIGHTS = (6, 3, 1)


def send(url, models, max_tokens, rng):
    model = rng.choice(models)
    words = rng.choice(INPUT_WORDS)
    mode = rng.choices(MODES, MODE_WEIGHTS)[0]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello " * words}],
        "max_tokens": rng.randint(1, max_tokens),
        "stream": mode != "plain",
    }
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start, code, error = time.time(), 0, ""
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            code = response.status
            if mode == "abort":
                response.readline()
            else:
                response.read()
    except urllib.error.HTTPError as http_error:
        code, error = http_error.code, http_error.read()[:200].decode(errors="replace")
    except Exception as other:
        error = repr(other)[:200]
    return {"t": start, "seconds": time.time() - start, "model": model, "words": words,
            "mode": mode, "code": code, "error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    lock = threading.Lock()
    deadline = time.time() + args.duration
    with open(args.output, "w") as output:
        def client(seed):
            rng = random.Random(seed)
            while time.time() < deadline:
                row = send(args.url, args.models, args.max_tokens, rng)
                with lock:
                    output.write(json.dumps(row) + "\n")
                    output.flush()

        with ThreadPoolExecutor(args.concurrency) as pool:
            list(pool.map(client, range(args.concurrency)))


if __name__ == "__main__":
    main()
