#!/usr/bin/env python3
"""Update an existing results JSON by filling `time_before_optim` and `time_after_optim`
from two `extract_real_times.py`-style JSON files.

Usage:
  python3 update_results_with_times.py --results results/results.json --before before.json --after after.json --output results/updated_results.json

The script expects the extract JSONs to have structure: { dataset: { caseN: {"seconds": float, ...}}}
Mapping: `scenario` in `results.json` matches dataset; `case_id` -> `case{case_id}`.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_seconds_from_extract(extract: Dict[str, Any], scenario: str, case_id: int):
    key = f'case{case_id}'
    ds = extract.get(scenario)
    if not isinstance(ds, dict):
        return None
    entry = ds.get(key)
    if not isinstance(entry, dict):
        return None
    return entry.get('seconds')


def update_results(results_path: str, before_path: str, after_path: str, output_path: str) -> None:
    results = load_json(results_path)
    before = load_json(before_path)
    after = load_json(after_path)

    updated = []
    total = 0
    updated_count = 0
    for item in results:
        total += 1
        scenario = item.get('scenario')
        case_id = item.get('case_id')
        if scenario is None or case_id is None:
            updated.append(item)
            continue
        b = get_seconds_from_extract(before, scenario, case_id)
        a = get_seconds_from_extract(after, scenario, case_id)
        # Only overwrite if we found a valid number; otherwise keep existing
        if isinstance(b, (int, float)):
            item['time_before_optim'] = float(b)
        if isinstance(a, (int, float)):
            item['time_after_optim'] = float(a)
        if isinstance(b, (int, float)) or isinstance(a, (int, float)):
            updated_count += 1
        updated.append(item)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f'Wrote {output_path}: updated {updated_count}/{total} entries')


def main() -> None:
    parser = argparse.ArgumentParser(description='Update results JSON with before/after timings from extract JSONs')
    parser.add_argument('--results', '-r', required=True, help='Path to existing results JSON (e.g. results/results.json)')
    parser.add_argument('--before', '-b', required=True, help='Extract JSON to use as time_before_optim')
    parser.add_argument('--after', '-a', required=True, help='Extract JSON to use as time_after_optim')
    parser.add_argument('--output', '-o', default=None, help='Output path (default: overwrite results file)')
    args = parser.parse_args()

    out = args.output or args.results
    update_results(args.results, args.before, args.after, out)


if __name__ == '__main__':
    main()
