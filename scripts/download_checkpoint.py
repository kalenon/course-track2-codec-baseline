#!/usr/bin/env python3
"""Download a course-hosted reference checkpoint.

The URL is supplied by the course release channel; no source competition URL
or machine-specific path is embedded in this repository.
"""

import argparse
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        if not args.output.is_file():
            raise FileExistsError(f"Output path is not a file: {args.output}")
        if args.output.stat().st_size > 0:
            print(f"Already downloaded: {args.output}")
            return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".download")
    try:
        with urllib.request.urlopen(args.url, timeout=60) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Downloaded file is empty: {temporary}")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded: {args.output}")


if __name__ == "__main__":
    main()
