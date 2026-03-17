#!/usr/bin/env python3
"""
Download and install argostranslate language packages.

Usage:
    python scripts/install_argos_packages.py
    python scripts/install_argos_packages.py --from vi --to en
    python scripts/install_argos_packages.py --from en --to vi

Default: vi→en and en→vi (both directions for bidirectional support)
"""
from __future__ import annotations

import argparse
import sys


def install_package(from_code: str, to_code: str) -> None:
    try:
        import argostranslate.package as pkg  # type: ignore[import]
    except ImportError:
        print("ERROR: argostranslate not installed. Run: pip install argostranslate")
        sys.exit(1)

    print(f"Updating package index…")
    pkg.update_package_index()

    available = pkg.get_available_packages()
    target = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )

    if target is None:
        print(f"ERROR: No package found for {from_code}→{to_code}")
        print("Available pairs:")
        for p in sorted(available, key=lambda x: (x.from_code, x.to_code)):
            print(f"  {p.from_code} → {p.to_code}")
        return

    print(f"Downloading {from_code}→{to_code} package…")
    path = target.download()
    pkg.install_from_path(path)
    print(f"Installed: {from_code}→{to_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_code", default=None)
    parser.add_argument("--to", dest="to_code", default=None)
    args = parser.parse_args()

    if args.from_code and args.to_code:
        install_package(args.from_code, args.to_code)
    else:
        # Default: install both directions for vi↔en
        install_package("vi", "en")
        install_package("en", "vi")


if __name__ == "__main__":
    main()
