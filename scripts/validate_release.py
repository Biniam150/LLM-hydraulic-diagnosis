"""Validate the staged files, benchmark sizes, and manuscript headline metrics."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]

def rows(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checksums',action='store_true',help='Check the release baseline; regeneration can change formatting and figure bytes.')
    args=parser.parse_args()
    problems=[]
    for path in ROOT.rglob('*.py'):
        try:ast.parse(path.read_text(encoding='utf-8-sig'))
        except SyntaxError as exc:problems.append(str(exc))
    for net,n in [('anytown',18),('net3',21),('bwnii',21)]:
        index=rows(f'data/benchmark/{net}/case_index.csv')
        if len(index)!=n:problems.append(f'{net}: expected {n} cases, got {len(index)}')
        for row in index:
            path=ROOT/row['observations']
            if not path.exists():problems.append(f'Missing observations {path}')
            elif len(rows(row['observations']))!=24:problems.append(f'{path}: expected 24 samples')
        profile=json.loads((ROOT/f'configs/networks/{net}.json').read_text())
        if not (ROOT/profile['inp_file']).exists():problems.append(f'Missing network for {net}')
    expected={'Anytown':13/18,'Net3':13/21,'BWN-II':16/21}
    for row in rows('results/experiment_3/generalized_verifier_v4_three_networks/generalized_verifier_three_networks.csv'):
        if abs(float(row['exact_outcome_accuracy'])-expected[row['network']])>1e-12:problems.append('Accuracy mismatch: '+row['network'])
        if float(row['accepted_diagnosis_precision'])!=1:problems.append('Precision mismatch: '+row['network'])
    for row in rows('results/experiment_4/experiment4_three_networks/experiment4_three_network_pooled_summary.csv'):
        if int(row['base_cases'])!=20 or int(row['context_variants'])!=80:problems.append('Context denominators mismatch')
    if args.checksums:
        for row in rows('results/release_inventory.csv'):
            p=ROOT/row['path']
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:problems.append('Checksum mismatch: '+row['path'])
    for p in ROOT.rglob('*'):
        if p.is_file() and p.stat().st_size>=100_000_000:problems.append('Large Git file: '+p.relative_to(ROOT).as_posix())
    if problems:raise SystemExit('\n'.join(problems))
    print('PASS: Python syntax, 60 benchmark cases, network paths, headline metrics, context denominators, and file sizes'+(', release checksums' if args.checksums else ''))

if __name__=='__main__':main()
