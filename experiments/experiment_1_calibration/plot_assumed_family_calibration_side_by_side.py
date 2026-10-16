from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from experiments.experiment_1_calibration.plot_assumed_family_calibration_comparison import COLORS, DEMAND_RESULTS, ROUGHNESS_RESULTS, _panel
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = ROOT / 'results' / '../figures/main' / 'anytown_assumed_family_calibration_side_by_side_revised'

def main() -> None:
    roughness = pd.read_csv(ROUGHNESS_RESULTS)
    demand = pd.read_csv(DEMAND_RESULTS)
    for frame in (roughness, demand):
        assert set(frame['seed']) == {101, 202, 404}
        assert frame.groupby('scenario_id').size().eq(3).all()
    roughness_order = ['true_roughness_control', 'demand_mismatch', 'source_head_mismatch', 'leak_mismatch', 'sensor_bias_mismatch', 'no_fault_control']
    roughness_labels = {'true_roughness_control': 'Roughness error', 'demand_mismatch': 'Demand error', 'source_head_mismatch': 'Source-head error', 'leak_mismatch': 'Leak', 'sensor_bias_mismatch': 'Sensor bias', 'no_fault_control': 'No fault'}
    demand_order = ['true_demand_control', 'roughness_mismatch', 'source_head_mismatch', 'leak_mismatch', 'sensor_bias_mismatch', 'no_fault_control']
    demand_labels = {'true_demand_control': 'Demand error', 'roughness_mismatch': 'Pipe-roughness error', 'source_head_mismatch': 'Source-head error', 'leak_mismatch': 'Leak', 'sensor_bias_mismatch': 'Sensor bias', 'no_fault_control': 'No fault'}
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    _panel(axes[0], roughness, roughness_order, roughness_labels, 'roughness_error', (-20, 85), '(a) Single-pipe roughness-only calibration')
    _panel(axes[1], demand, demand_order, demand_labels, 'demand_error', (-225, 115), '(b) Single-node demand-only calibration')
    legend = [Patch(facecolor=COLORS['matched'], label='Same-family scenario'), Patch(facecolor=COLORS['mismatched'], label='Different-family scenario'), Patch(facecolor=COLORS['no_fault'], label='No fault'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor='#404040', markersize=6, label='Individual noise realization (n = 3)')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.77, bottom=0.28, wspace=0.36)
    fig.suptitle('Anytown: six controlled scenarios', y=0.99, fontsize=13)
    for ax in axes:
        ax.set_xlabel('Full-window pressure MAE reduction (%)')
    fig.text(0.5, 0.045, 'Bars: mean over seeds 101, 202, 404. Positive = lower error; negative = higher error.\nDiscrete fitting on even hours; plotted MAE uses all 24 hourly observations (0–23 h).', ha='center', va='bottom', fontsize=9, color='#444444')
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_STEM.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(OUTPUT_STEM.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
