#!/usr/bin/env python3
"""Build final comparison JSON from saved/baseline run directories.

Rules:
1. For `time_after_optim`, choose the minimum `real` time among saved roots.
2. For `time_before_optim`, choose the maximum `real` time among baseline roots.
3. PSNR is read from one specified root (`--psnr-root`) per scenario/case.
4. Optionally save a speedup JSON (`time_before_optim / time_after_optim`).

Expected directory layout:
  <root>/<scenario>/case<id>/output.log
  <psnr_root>/<scenario>/case<id>/psnr_result.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_SCENARIOS = [
    "unitree_z1_stackbox",
    "unitree_z1_dual_arm_stackbox",
    "unitree_z1_dual_arm_stackbox_v2",
    "unitree_z1_dual_arm_cleanup_pencils",
    "unitree_g1_pack_camera",
]

DEFAULT_CASE_IDS = [1, 2, 3, 4]

DURATION_RE = re.compile(r"(?:(?P<h>\d+)h)?\s*(?:(?P<m>\d+)m)?\s*(?P<s>\d+(?:\.\d+)?)s")
REAL_LINE_RE = re.compile(r"\breal\b\s*(.*)")
PSNR_LINE_RE = re.compile(r"Video\s+PSNR:\s*([-+]?\d+(?:\.\d+)?)")


def parse_duration_to_seconds(text: str) -> Optional[float]:
    """Parse durations such as '3m29.315s' into seconds."""
    m = DURATION_RE.search(text.strip())
    if not m:
        return None
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m") or 0)
    seconds = float(m.group("s"))
    return hours * 3600 + minutes * 60 + seconds


def read_real_seconds(output_log: Path) -> float:
    """Extract the last valid `real` duration from an output log."""
    if not output_log.is_file():
        raise FileNotFoundError(f"Missing output.log: {output_log}")

    last_seconds: Optional[float] = None
    with output_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "real" not in line:
                continue
            m = REAL_LINE_RE.search(line)
            if not m:
                continue
            seconds = parse_duration_to_seconds(m.group(1))
            if seconds is not None:
                last_seconds = seconds

    if last_seconds is None:
        raise ValueError(f"No valid `real` time found in: {output_log}")
    return last_seconds


def read_psnr(psnr_root: Path, scenario: str, case_id: int) -> Optional[float]:
    """Read PSNR from psnr_result.json, fallback to output.log if needed."""
    case_dir = psnr_root / scenario / f"case{case_id}"
    psnr_json = case_dir / "psnr_result.json"
    if psnr_json.is_file():
        with psnr_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = data.get("psnr")
        if isinstance(value, (int, float)):
            return float(value)

    output_log = case_dir / "output.log"
    if output_log.is_file():
        with output_log.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = PSNR_LINE_RE.search(line)
                if m:
                    return float(m.group(1))
    return None


def pick_time_across_roots(
    roots: Iterable[Path],
    scenario: str,
    case_id: int,
    mode: str,
) -> Tuple[float, Path]:
    """Pick min/max time across roots for one scenario/case."""
    candidates: List[Tuple[float, Path]] = []
    for root in roots:
        output_log = root / scenario / f"case{case_id}" / "output.log"
        if not output_log.is_file():
            continue
        seconds = read_real_seconds(output_log)
        candidates.append((seconds, root))

    if not candidates:
        roots_text = ", ".join(str(r) for r in roots)
        raise FileNotFoundError(
            f"No valid output.log found for {scenario}/case{case_id} under: {roots_text}"
        )

    if mode == "min":
        return min(candidates, key=lambda x: x[0])
    if mode == "max":
        return max(candidates, key=lambda x: x[0])
    raise ValueError(f"Unsupported mode: {mode}")


def validate_roots(roots: Iterable[Path], arg_name: str) -> None:
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"Invalid {arg_name}: {missing}")


def build_results(
    saved_roots: List[Path],
    baseline_roots: List[Path],
    psnr_root: Path,
    scenarios: List[str],
    case_ids: List[int],
    psnr_round: Optional[int],
    allow_missing_psnr: bool,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for scenario in scenarios:
        for case_id in case_ids:
            time_after, _ = pick_time_across_roots(saved_roots, scenario, case_id, mode="min")
            time_before, _ = pick_time_across_roots(
                baseline_roots, scenario, case_id, mode="max"
            )

            psnr = read_psnr(psnr_root, scenario, case_id)
            if psnr is None and not allow_missing_psnr:
                raise ValueError(
                    f"Missing PSNR for {scenario}/case{case_id} in {psnr_root}. "
                    "Expected psnr_result.json or a PSNR line in output.log."
                )
            if psnr is not None and psnr_round is not None and psnr_round >= 0:
                psnr = round(psnr, psnr_round)

            results.append(
                {
                    "scenario": scenario,
                    "case_id": case_id,
                    "time_before_optim": time_before,
                    "time_after_optim": time_after,
                    "psnr": psnr,
                }
            )
    return results


def build_speedup_results(
    results: List[Dict[str, object]],
    speedup_round: Optional[int],
) -> List[Dict[str, object]]:
    """Build per-case speedup entries from final results."""
    speedup_data: List[Dict[str, object]] = []
    for item in results:
        before = item.get("time_before_optim")
        after = item.get("time_after_optim")

        speedup: Optional[float] = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and after != 0:
            speedup = float(before) / float(after)
            if speedup_round is not None and speedup_round >= 0:
                speedup = round(speedup, speedup_round)

        speedup_data.append(
            {
                "scenario": item.get("scenario"),
                "case_id": item.get("case_id"),
                "time_before_optim": before,
                "time_after_optim": after,
                "speedup": speedup,
            }
        )
    return speedup_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scenario/case JSON by min(saved time), max(baseline time), and PSNR root."
    )
    parser.add_argument(
        "--saved-roots",
        nargs="+",
        default=["saved_6", "saved_7", "saved_8", "saved_9"],
        help="Saved roots for choosing minimum `real` time.",
    )
    parser.add_argument(
        "--baseline-roots",
        nargs="+",
        default=["baseline_1", "baseline_2"],
        help="Baseline roots for choosing maximum `real` time.",
    )
    parser.add_argument(
        "--psnr-root",
        required=True,
        help="Single root used to read PSNR per scenario/case.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario names in output order.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=DEFAULT_CASE_IDS,
        help="Case IDs in output order.",
    )
    parser.add_argument(
        "--psnr-round",
        type=int,
        default=4,
        help="Round PSNR to N decimals. Use negative value to disable rounding.",
    )
    parser.add_argument(
        "--allow-missing-psnr",
        action="store_true",
        help="Allow missing PSNR (write null instead of raising error).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/updated_results_from_runs.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--speedup-output",
        default=None,
        help=(
            "Optional speedup JSON output path. "
            "Default: <output_stem>_speedup.json in the same directory."
        ),
    )
    parser.add_argument(
        "--speedup-round",
        type=int,
        default=4,
        help="Round speedup to N decimals. Use negative value to disable rounding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    saved_roots = [Path(root) for root in args.saved_roots]
    baseline_roots = [Path(root) for root in args.baseline_roots]
    psnr_root = Path(args.psnr_root)
    out_path = Path(args.output)

    validate_roots(saved_roots, "--saved-roots")
    validate_roots(baseline_roots, "--baseline-roots")
    if not psnr_root.is_dir():
        raise FileNotFoundError(f"Invalid --psnr-root: {psnr_root}")

    psnr_round = None if args.psnr_round < 0 else args.psnr_round
    speedup_round = None if args.speedup_round < 0 else args.speedup_round

    results = build_results(
        saved_roots=saved_roots,
        baseline_roots=baseline_roots,
        psnr_root=psnr_root,
        scenarios=args.scenarios,
        case_ids=args.case_ids,
        psnr_round=psnr_round,
        allow_missing_psnr=args.allow_missing_psnr,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    speedup_out_path = (
        Path(args.speedup_output)
        if args.speedup_output
        else out_path.with_name(f"{out_path.stem}_speedup.json")
    )
    speedup_out_path.parent.mkdir(parents=True, exist_ok=True)
    speedup_data = build_speedup_results(results, speedup_round=speedup_round)
    with speedup_out_path.open("w", encoding="utf-8") as f:
        json.dump(speedup_data, f, indent=2, ensure_ascii=False)

    valid_speedups = [
        item["speedup"] for item in speedup_data if isinstance(item.get("speedup"), (int, float))
    ]
    avg_speedup = (sum(valid_speedups) / len(valid_speedups)) if valid_speedups else None

    print(f"Wrote {len(results)} entries to {out_path}")
    print(f"Wrote {len(speedup_data)} speedup entries to {speedup_out_path}")
    if avg_speedup is not None:
        print(f"Average speedup: {avg_speedup:.4f} ({len(valid_speedups)} cases)")
    else:
        print("Average speedup: N/A (no valid cases)")


if __name__ == "__main__":
    main()
