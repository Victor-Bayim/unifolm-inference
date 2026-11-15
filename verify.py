#!/usr/bin/env python3
"""Recompute archived timing results. Python standard library only; no GPU run."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, math, re, statistics, sys

ROOT=Path(__file__).resolve().parent

def load_module(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def recompute():
    baseline=json.loads((ROOT/'benchmarks/baseline.json').read_text())
    optimized=json.loads((ROOT/'benchmarks/ours.json').read_text())
    saved=json.loads((ROOT/'benchmarks/speedup_real_vs_merged.json').read_text())
    assert set(baseline)==set(optimized)==set(saved), 'Scenario sets differ'
    parser=load_module('original_parser','project/build_results_json_from_runs.py')
    original=load_module('original_ratios','project/compute_speedup.py')
    calculated=original.compute_speedup(baseline,optimized)
    rows=[]
    for scene in sorted(baseline):
        assert set(baseline[scene])==set(optimized[scene])=={'case1','case2','case3','case4'}
        for case in sorted(baseline[scene]):
            a,b=baseline[scene][case],optimized[scene][case]
            for item in [a,b]:
                assert item['seconds']>0 and math.isfinite(item['seconds'])
                assert math.isclose(parser.parse_duration_to_seconds(item['raw']),item['seconds'],abs_tol=1e-8), 'Duration text differs from stored seconds'
            ratio=a['seconds']/b['seconds']
            assert math.isclose(ratio,saved[scene][case]['speedup'],rel_tol=1e-12), 'Archived ratio mismatch'
            assert math.isclose(ratio,calculated[scene][case]['speedup'],rel_tol=1e-12)
            rows.append({'scenario':scene,'case':case,'baseline_seconds':a['seconds'],'optimized_seconds':b['seconds'],'speedup':ratio,'baseline_run':a['path'].split('/')[0],'optimized_run':b['path'].split('/')[0]})
    with (ROOT/'benchmarks/cases.csv').open(encoding='utf-8',newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    assert len(rows)==len(csv_rows)==20
    for a,b in zip(rows,csv_rows):
        for key in a:
            assert math.isclose(a[key],float(b[key]),rel_tol=1e-12) if isinstance(a[key],float) else a[key]==b[key], 'CSV differs from archived JSON'
    before=statistics.mean(x['baseline_seconds'] for x in rows)
    after=statistics.mean(x['optimized_seconds'] for x in rows)
    return {'cases':len(rows),'baseline_mean_seconds':before,'optimized_mean_seconds':after,'ratio_of_means':before/after,'mean_of_case_ratios':statistics.mean(x['speedup'] for x in rows)}

def integrity():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    maintenance=json.loads((ROOT/'maintenance.json').read_text())
    maintained={item['path']:item for item in maintenance['files']}
    assert len(maintained)==len(maintenance['files']), 'Duplicate maintenance path'
    historical={item['path']:item for item in manifest['files']}
    for path, change in maintained.items():
        assert path in historical, 'Maintenance path is absent from original manifest'
        assert change['original_sha256']==historical[path]['sha256'], 'Historical hash differs'
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==change['maintained_sha256'], 'Maintained source changed: '+path
        assert hashlib.sha256((ROOT/change['patch']).read_bytes()).hexdigest()==change['patch_sha256'], 'Maintenance patch changed'
    for item in manifest['files']:
        change=maintained.get(item['path'])
        original_path=change['original_copy'] if change else item['path']
        data=(ROOT/original_path).read_bytes()
        assert hashlib.sha256(data).hexdigest()==item['sha256'], 'Historical content changed: '+item['path']
    python_files=list((ROOT/'project').rglob('*.py'))
    for file in python_files:
        ast.parse(file.read_text(encoding='utf-8'),filename=str(file.relative_to(ROOT)))
    return {'verified_archive_files':len(manifest['files']),'python_files_parsed':len(python_files),'maintained_files':len(maintained)}

def main():
    result=recompute()
    stored=json.loads((ROOT/'benchmarks/summary.json').read_text())
    for key,value in result.items():
        assert math.isclose(value,stored[key],rel_tol=1e-12), 'Summary differs: '+key
    result.update(integrity())
    result['validation_scope']='Archived data recomputation, original/maintained source integrity and syntax; no GPU inference performed'
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
