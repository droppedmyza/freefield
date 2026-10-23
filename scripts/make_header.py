#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freefield.header_card import SIZE, STYLES, write_header  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a header card (toy box or magazine masthead) for the clip.")
    ap.add_argument("title")
    ap.add_argument("out", type=Path)
    ap.add_argument("--style", choices=sorted(STYLES), default="toy")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--card", help="card color (default from the style)")
    ap.add_argument("--fill", help="letter color (default from the style)")
    ap.add_argument("--edge", help="bottom edge / subtitle accent color (default from the style)")
    a = ap.parse_args()
    if not a.title.strip():
        ap.error("title is empty")
    write_header(a.title, a.out, a.style, a.subtitle, card=a.card, fill=a.fill, edge=a.edge)
    print(f"wrote {a.out} ({SIZE[0]}x{SIZE[1]})")


if __name__ == "__main__":
    main()
