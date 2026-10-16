"""Recompute final summaries from saved decisions; no LLM calls or fitting."""
from pathlib import Path
import argparse
import csv
import json
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def calibration_summary():
    rows = []
    for family in ['roughness', 'demand']:
        source = ROOT/f'results/experiment_1/anytown_false_{family}_calibration/false_{family}_results.csv'
        with source.open(encoding='utf-8-sig') as f:
            data = list(csv.DictReader(f))
        for scenario in dict.fromkeys(x['scenario_id'] for x in data):
            group = [x for x in data if x['scenario_id']==scenario]
            rows.append({'assumed_family':family,'scenario_id':scenario,'runs':len(group),
                'mean_pressure_mae_reduction_percent':sum(float(x['pressure_mae_improvement'])*100 for x in group)/len(group)})
    with (ROOT/'results/experiment_1/calibration_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['all','1','2','3','4'],default='all')
    args=parser.parse_args()
    os.environ.setdefault('MPLBACKEND','Agg')
    os.chdir(ROOT)
    selected=['1','2','3','4'] if args.experiment=='all' else [args.experiment]
    modules={
        '2':'experiments.experiment_2_residual_diagnosis.summarize_experiment2_three_networks',
        '3':'experiments.experiment_3_hydraulic_verification.summarize_experiment3_three_networks',
        '4':'experiments.experiment_4_context_followup.summarize_experiment4_three_networks',
    }
    for experiment in selected:
        print('Reproducing Experiment',experiment,flush=True)
        if experiment=='1':calibration_summary()
        else:
            sys.argv=[modules[experiment]]
            runpy.run_module(modules[experiment],run_name='__main__')


if __name__=='__main__':main()
