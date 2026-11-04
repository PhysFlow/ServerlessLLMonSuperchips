from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from dataclasses import replace

from .application import ControlPlane
from .config import ConfigError, load_config
from .errors import ControlPlaneError

__all__ = ["main"]


def _forward_to_sigint(signum, frame) -> None:
    os.kill(os.getpid(), signal.SIGINT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="memserve-control-plane", description="Start the MemServe Control Plane",
    )
    parser.add_argument("--config", required=True, help="path to the user configuration file")
    parser.add_argument("--host", help="override the configured gateway host")
    parser.add_argument("--port", type=int, help="override the configured gateway port")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    if args.host is not None:
        config = replace(config, gateway=replace(config.gateway, host=args.host))
    if args.port is not None:
        config = replace(config, gateway=replace(config.gateway, port=args.port))

    signal.signal(signal.SIGHUP, _forward_to_sigint)
    signal.signal(signal.SIGTERM, _forward_to_sigint)
    try:
        asyncio.run(ControlPlane(config).run())
    except KeyboardInterrupt:
        pass
    except (ControlPlaneError, RuntimeError, ValueError) as error:
        print(f"startup error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
