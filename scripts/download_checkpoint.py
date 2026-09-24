#!/usr/bin/env python3
"""Download a course-hosted reference checkpoint with SHA256 verification.

The URL is supplied by the course release channel; no source competition URL
or machine-specific path is embedded in this repository.
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    expected = args.sha256.lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        parser.error("--sha256 must be a 64-character hexadecimal digest")
    if args.output.exists() and digest(args.output) == expected:
        print(f"Already verified: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".download")
    try:
        with urllib.request.urlopen(args.url, timeout=60) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        actual = digest(temporary)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: expected {expected}, got {actual}")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified: {args.output}")


if __name__ == "__main__":
    main()
