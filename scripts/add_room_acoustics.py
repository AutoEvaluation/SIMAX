#!/usr/bin/env python3
"""Add room acoustics to synthesized dialogues."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simax.audio import wet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    args = ap.parse_args(argv)
    for dry in sorted(Path(args.root).glob("*/dialogue_full.wav")):
        out = dry.with_name("dialogue_wet.wav")
        if out.exists() and out.stat().st_mtime >= dry.stat().st_mtime:
            continue
        seed = int(hashlib.sha256(dry.parent.name.encode()).hexdigest()[:8], 16) % 100000
        wet.main(["--dry", str(dry), "--out", str(out), "--seed", str(seed)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
