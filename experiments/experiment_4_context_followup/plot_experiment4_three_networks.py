from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
PROJECT = Path(__file__).resolve().parents[2]
INPUT = PROJECT / 'results/experiment_4/experiment4_three_networks/experiment4_three_network_model_summary.csv'
OUTPUT = PROJECT / 'figures/main/experiment4_three_networks'
MODELS = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B']
NETWORKS = ['Anytown', 'Net3', 'BWN-II']
COUNTS = {'Anytown': 5, 'Net3': 11, 'BWN-II': 4}

def panel(ax: plt.Axes, data: pd.DataFrame, llm_column: str, hybrid_column: str, reference: float, show_y: bool) -> None:
    ordered = data.set_index('model_label').reindex(MODELS)
    y = np.arange(len(MODELS))
    llm = 100 * ordered[llm_column].to_numpy(float)
    hybrid = 100 * ordered[hybrid_column].to_numpy(float)
    for yi, x0, x1 in zip(y, llm, hybrid):
        ax.plot([x0, x1], [yi, yi], color='#C5C5C5', linewidth=1.8, zorder=1)
    ax.scatter(llm, y, s=62, marker='o', facecolor='white', edgecolor='#7A7A7A', linewidth=1.7, zorder=3)
    ax.scatter(hybrid, y, s=68, marker='D', color='#2E8B76', edgecolor='white', linewidth=0.8, zorder=4)
    ax.axvline(reference, color='#B07A3A', linestyle='--', linewidth=1.4, zorder=0)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS if show_y else [])
    ax.invert_yaxis()
    ax.grid(axis='x', color='#E4E4E4', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

def main() -> None:
    data = pd.read_csv(INPUT)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5, 'axes.labelsize': 12.5, 'axes.titlesize': 13.5, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True)
    letters = iter('abcdef')
    for col, network in enumerate(NETWORKS):
        subset = data[data['network'].eq(network)]
        panel(axes[0, col], subset, 'helpful_llm_only_resolution', 'helpful_verified_resolution', reference=0, show_y=col == 0)
        axes[0, col].set_title(f'{network} (n = {COUNTS[network]})', pad=10)
        axes[0, col].text(-0.12, 1.04, f'({next(letters)})', transform=axes[0, col].transAxes, fontsize=12)
        panel(axes[1, col], subset, 'nonhelpful_llm_only_safety', 'nonhelpful_verified_safety', reference=100, show_y=col == 0)
        axes[1, col].text(-0.12, 1.04, f'({next(letters)})', transform=axes[1, col].transAxes, fontsize=12)
        axes[1, col].set_xlabel('Cases (%)')
    axes[0, 0].set_ylabel('Helpful-context resolution')
    axes[1, 0].set_ylabel('Non-helpful-context safety')
    legend = [Line2D([0], [0], marker='o', linestyle='', markerfacecolor='white', markeredgecolor='#7A7A7A', markeredgewidth=1.7, markersize=8, label='LLM context, unverified'), Line2D([0], [0], marker='D', linestyle='', markerfacecolor='#2E8B76', markeredgecolor='white', markersize=8, label='Verified hybrid'), Line2D([0], [0], color='#B07A3A', linestyle='--', linewidth=1.5, label='Hydraulic-only reference')]
    fig.legend(handles=legend, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.16, right=0.985, hspace=0.32, wspace=0.16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(OUTPUT.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
if __name__ == '__main__':
    main()
