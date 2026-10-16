"""Regenerate frozen synthetic observations from the retained final case IDs."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, generate_pseudo_observations


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    provenance = []
    for network, design_name in [('anytown', 'anytown_causal_benchmark'), ('net3', 'net3_causal_replication'), ('bwnii', 'bwnii_generalized_external_test')]:
        profile = load_network_profile(ROOT / f'configs/networks/{network}.json')
        design = json.loads((ROOT / f'configs/scenarios/{design_name}.json').read_text())
        scenarios = {row['id']: row['fault'] for row in design['scenarios']}
        run = ROOT / f'results/experiment_3/experiment3_{network}_generalized_verifier_v4/hybrid_v3_results.csv'
        with run.open(encoding='utf-8-sig') as stream:
            source_rows = list(csv.DictReader(stream))
        cases = {(x['scenario_id'], int(x['observation_seed'])): x for x in source_rows}
        index, truth = [], []
        for number, ((scenario, seed), row) in enumerate(cases.items(), 1):
            case_id = f'case_{number:03d}'
            fault = FaultSpec(**scenarios[scenario])
            config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link,
                pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction,
                duration_hours=profile.duration_hours, seed=seed, solver=profile.solver, demand_model=profile.demand_model,
                feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)
            observed, _, _ = generate_pseudo_observations(ROOT / profile.inp_file, fault, config)
            relative = f'data/benchmark/{network}/observations/{case_id}.csv'
            output = ROOT / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            observed.to_csv(output, index_label='hour')
            index.append({'case_id':case_id, 'observations':relative, 'seed':seed, 'samples':len(observed)})
            truth.append({'case_id':case_id,'source_neutral_case_id':row['neutral_case_id'],'scenario_id':scenario,'seed':seed,
                'true_cause':fault.fault_type,'true_target':fault.target,'magnitude':fault.magnitude,
                'start_hour':fault.start_hour,'end_hour':fault.end_hour,'expected_outcome':row['expected_label']})
            provenance.append({'path':relative,'kind':'regenerated_synthetic_observations',
                'source':f'configs/scenarios/{design_name}.json; {run.relative_to(ROOT).as_posix()}',
                'generator':'python scripts/export_benchmark.py','sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
            print(network, case_id, scenario, flush=True)
        assert len(index) == (18 if network=='anytown' else 21)
        write_csv(ROOT/f'data/benchmark/{network}/case_index.csv',index)
        write_csv(ROOT/f'data/benchmark/{network}/evaluation_only/ground_truth.csv',truth)
        write_csv(ROOT/f'docs/supplement/{network}_scenarios.csv',truth)
    write_csv(ROOT/'data/provenance.csv',provenance)


if __name__ == '__main__':
    main()
