"""Regenerate the five numerical main figures from retained result tables."""
from pathlib import Path
import os
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND','Agg')
    modules=[
        'experiments.experiment_1_calibration.plot_assumed_family_calibration_large_labels',
        'experiments.experiment_2_residual_diagnosis.plot_experiment2_three_networks_final',
        'experiments.experiment_2_residual_diagnosis.plot_experiment2_performance_tokens_graphical',
        'experiments.experiment_3_hydraulic_verification.plot_experiment3_reliability_absolute_scores',
        'experiments.experiment_3_hydraulic_verification.plot_experiment3_network_line_comparison',
        'experiments.sensitivity.plot_verifier_sensitivity_grouped_boxes',
        'experiments.experiment_4_context_followup.plot_experiment4_three_networks',
    ]
    for module in modules:
        print('Rendering',module,flush=True)
        runpy.run_module(module,run_name='__main__')
