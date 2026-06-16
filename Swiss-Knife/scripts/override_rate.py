"""Parse Swiss Knife --verbose logs and report blade-override rates.

For every tournament match the verbose logger emits a line like:

    Round 1 | Match: c0 vs c3  Δdraft=0.1234  Δblade=-0.5678  score=-0.1234 → winner=c3

"Override" = match where sign(Δdraft) ≠ sign(final score), i.e. the blade
swung the winner away from the draft's preference. This is the closest
direct measurement of "how often did alignment guidance actually do
something" available without modifying the generator.

Usage:
    python scripts/override_rate.py runs/<dir>/*.txt
    python scripts/override_rate.py file1.txt file2.txt ...
"""

import glob
import os
import re
import sys

# Use \u escapes so the file stays ASCII-clean regardless of terminal encoding.
PAT = re.compile(
    r"Δdraft=(-?[\d.]+)\s+Δblade=(-?[\d.]+)\s+score=(-?[\d.]+)"
)


def expand(args):
    files = []
    for a in args:
        if any(c in a for c in "*?["):
            files.extend(sorted(glob.glob(a)))
        elif os.path.isfile(a):
            files.append(a)
    return files


def analyze(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    matches = PAT.findall(text)
    if not matches:
        return None
    n = len(matches)
    flipped = sum(
        1 for d, _b, s in matches if (float(d) > 0) != (float(s) > 0)
    )
    return n, flipped


def main():
    files = expand(sys.argv[1:])
    if not files:
        print("No files matched.")
        sys.exit(1)

    print(f"{'File':<55} {'Matches':>8} {'Flipped':>8} {'Rate':>7}")
    print("-" * 80)
    total_m, total_f = 0, 0
    for f in files:
        result = analyze(f)
        name = os.path.basename(f)
        if result is None:
            print(f"{name:<55} {'-':>8} {'-':>8} {'-':>7}")
            continue
        n, flipped = result
        rate = 100 * flipped / n
        print(f"{name:<55} {n:>8d} {flipped:>8d} {rate:>6.1f}%")
        total_m += n
        total_f += flipped
    if total_m > 0:
        print("-" * 80)
        print(
            f"{'TOTAL':<55} {total_m:>8d} {total_f:>8d} "
            f"{100*total_f/total_m:>6.1f}%"
        )


if __name__ == "__main__":
    main()
