#!/usr/bin/env python3
"""Compute speedup ratios between two JSON results produced by `extract_real_times.py`.

Usage:
  python3 compute_speedup.py ref.json other.json --output speedup.json

By default speedup = ref_seconds / other_seconds. If you prefer the inverse, swap inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_speedup(ref: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    datasets = set(ref.keys()) | set(other.keys())
    for ds in sorted(datasets):
        out[ds] = {}
        ref_cases = ref.get(ds) or {}
        other_cases = other.get(ds) or {}
        cases = set(ref_cases.keys()) | set(other_cases.keys())
        for case in sorted(cases):
            r = ref_cases.get(case)
            o = other_cases.get(case)
            r_secs = r.get('seconds') if isinstance(r, dict) else None
            o_secs = o.get('seconds') if isinstance(o, dict) else None
            entry = {
                'ref_seconds': r_secs,
                'other_seconds': o_secs,
                'ref_path': r.get('path') if isinstance(r, dict) else None,
                'other_path': o.get('path') if isinstance(o, dict) else None,
            }
            if r_secs is None or o_secs is None:
                entry['speedup'] = None
                entry['status'] = 'missing'
            else:
                # guard against zero
                if o_secs == 0:
                    entry['speedup'] = None
                    entry['status'] = 'invalid_other_zero'
                else:
                    entry['speedup'] = float(r_secs) / float(o_secs)
                    entry['status'] = 'ok'
            out[ds][case] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute speedup between two extracted JSON files.')
    parser.add_argument('ref', help='Reference JSON file (numerator)')
    parser.add_argument('other', help='Other JSON file (denominator)')
    parser.add_argument('--output', '-o', default='speedup.json', help='Output JSON path')
    args = parser.parse_args()

    ref = load_json(args.ref)
    other = load_json(args.other)
    out = compute_speedup(ref, other)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    total = 0
    ok = 0
    ratios = []
    for ds_vals in out.values():
        for case_vals in ds_vals.values():
            total += 1
            if case_vals.get('status') == 'ok' and case_vals.get('speedup') is not None and not math.isnan(case_vals.get('speedup')):
                ok += 1
                ratios.append(case_vals['speedup'])
    avg = float(sum(ratios) / len(ratios)) if ratios else None
    print(f'Wrote {args.output}: {ok}/{total} valid ratios', end='')
    if avg is not None:
        print(f', avg={avg:.4f}')
    else:
        print()


if __name__ == '__main__':
    main()
