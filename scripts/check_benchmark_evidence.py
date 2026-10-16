"""Check regenerated observations against saved nominal goodness-of-fit evidence."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import ObservationConfig, nominal_prediction
from experiments.experiment_3_hydraulic_verification.evaluate_anytown_hybrid_v3_adequacy import validation_gof

def main():
    reference=pd.read_csv(ROOT/'results/sensitivity/experiment3_verifier_sensitivity/physical_case_gof_evidence.csv')
    checked=0
    for key,name in [('anytown','Anytown'),('net3','Net3'),('bwnii','BWN-II')]:
        profile=load_network_profile(ROOT/f'configs/networks/{key}.json')
        config=ObservationConfig(pressure_sensors=list(profile.pressure_sensors),flow_link=profile.flow_link,
            pressure_noise_std_m=profile.pressure_noise_std_m,flow_noise_fraction=profile.flow_noise_fraction,
            duration_hours=profile.duration_hours,seed=1009,solver=profile.solver,demand_model=profile.demand_model,
            feature_policy=profile.feature_policy,search_policy=profile.search_policy,evidence_policy=profile.evidence_policy)
        nominal=nominal_prediction(ROOT/profile.inp_file,config)
        truth=pd.read_csv(ROOT/f'data/benchmark/{key}/evaluation_only/ground_truth.csv')
        for row in truth.itertuples(index=False):
            observed=pd.read_csv(ROOT/f'data/benchmark/{key}/observations/{row.case_id}.csv',index_col='hour')
            _,statistic,df=validation_gof(observed,nominal,nominal,config,list(range(1,24,2)))
            expected=reference[(reference.network==name)&(reference.scenario_id==row.scenario_id)&(reference.observation_seed==row.seed)].iloc[0]
            if df!=expected.nominal_gof_df or not np.isclose(statistic,expected.nominal_gof_statistic,rtol=1e-8,atol=1e-8):
                raise AssertionError(f'{key}/{row.case_id}: nominal evidence mismatch ({statistic}, {expected.nominal_gof_statistic})')
            checked+=1
    print(f'PASS: {checked} exported cases reproduce saved nominal goodness-of-fit evidence')

if __name__=='__main__':main()
