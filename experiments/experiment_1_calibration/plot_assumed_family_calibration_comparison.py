from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
ROUGHNESS_RESULTS = ROOT / 'results' / 'experiment_1/anytown_false_roughness_calibration' / 'false_roughness_results.csv'
DEMAND_RESULTS = ROOT / 'results' / 'experiment_1/anytown_false_demand_calibration' / 'false_demand_results.csv'
OUTPUT_STEM = ROOT / 'results' / '../figures/main' / 'anytown_assumed_family_calibration_comparison'
COLORS = {'matched': '#2E7D32', 'mismatched': '#F2C94C', 'no_fault': '#8D8D8D'}

def _panel(ax: plt.Axes, data: pd.DataFrame, order: list[str], labels: dict[str, str], assumed_cause: str, x_limits: tuple[float, float], panel_caption: str) -> None:
    means = data.groupby('scenario_id')['pressure_mae_improvement'].mean().reindex(order).mul(100.0)
    causes = data.groupby('scenario_id')['true_cause'].first().reindex(order)
    categories = ['no_fault' if cause == 'no_fault' else 'matched' if cause == assumed_cause else 'mismatched' for cause in causes]
    y = np.arange(len(order))
    bars = ax.barh(y, means.to_numpy(), height=0.58, color=[COLORS[item] for item in categories], zorder=2)
    for row_index, scenario_id in enumerate(order):
        observations = data.loc[data['scenario_id'] == scenario_id, 'pressure_mae_improvement'].to_numpy(dtype=float) * 100.0
        ax.scatter(observations, row_index + np.linspace(-0.11, 0.11, observations.size), s=22, facecolor='white', edgecolor='#404040', linewidth=0.75, zorder=3)
    span = x_limits[1] - x_limits[0]
    for bar, value, category in zip(bars, means.to_numpy(), categories):
        y_text = bar.get_y() + bar.get_height() / 2
        if value >= 0 and value >= 0.18 * span:
            x_text, ha = (value - 0.025 * span, 'right')
            color = '#202020' if category == 'mismatched' else 'white'
        elif value >= 0:
            x_text, ha, color = (value + 0.018 * span, 'left', '#202020')
        elif abs(value) >= 0.18 * span:
            x_text, ha = (value + 0.025 * span, 'left')
            color = '#202020' if category == 'mismatched' else 'white'
        else:
            x_text, ha, color = (value - 0.018 * span, 'right', '#202020')
        ax.text(x_text, y_text, f'{value:.1f}%', ha=ha, va='center', fontsize=9, color=color, zorder=4)
    ax.axvline(0, color='#555555', linewidth=0.9, zorder=1)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlim(*x_limits)
    ax.set_xlabel('Reduction in pressure MAE (%)')
    ax.grid(axis='x', color='#D7D7D7', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.25, panel_caption, transform=ax.transAxes, ha='center', va='top', fontsize=10)

def main() -> None:
    roughness = pd.read_csv(ROUGHNESS_RESULTS)
    demand = pd.read_csv(DEMAND_RESULTS)
    roughness_order = ['true_roughness_control', 'demand_mismatch', 'source_head_mismatch', 'leak_mismatch', 'sensor_bias_mismatch', 'no_fault_control']
    roughness_labels = {'true_roughness_control': 'True roughness error', 'demand_mismatch': 'Demand error', 'source_head_mismatch': 'Source-head error', 'leak_mismatch': 'Leak', 'sensor_bias_mismatch': 'Sensor bias', 'no_fault_control': 'No fault'}
    demand_order = ['true_demand_control', 'roughness_mismatch', 'source_head_mismatch', 'leak_mismatch', 'sensor_bias_mismatch', 'no_fault_control']
    demand_labels = {'true_demand_control': 'True demand error', 'roughness_mismatch': 'Pipe-roughness error', 'source_head_mismatch': 'Source-head error', 'leak_mismatch': 'Leak', 'sensor_bias_mismatch': 'Sensor bias', 'no_fault_control': 'No fault'}
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.6))
    _panel(axes[0], roughness, roughness_order, roughness_labels, 'roughness_error', (-20, 85), '(a) Single-pipe roughness-only calibration')
    _panel(axes[1], demand, demand_order, demand_labels, 'demand_error', (-225, 115), '(b) Single-node demand-only calibration')
    legend = [Patch(facecolor=COLORS['matched'], label='Matched cause'), Patch(facecolor=COLORS['mismatched'], label='Mismatched cause'), Patch(facecolor=COLORS['no_fault'], label='No fault'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor='#404040', markersize=6, label='Individual noise seed (n = 3)')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.08, hspace=0.67)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_STEM.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
