from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
base = ROOT / 'results'
df = pd.read_csv(base / 'experiment_2/experiment2_three_network_comparison/experiment2_three_network_summary.csv')
folders = {'Anytown': 'experiment_2/experiment2_anytown_classical_baselines', 'Net3': 'experiment_2/experiment2_net3_classical_baselines_current', 'BWN-II': 'experiment_2/experiment2_bwnii_classical_baselines'}
for net, folder in folders.items():
    row = pd.read_csv(base / folder / 'classical_baseline_summary.csv').query("method == 'Majority class'").iloc[0].to_dict()
    row['network'] = net
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
methods = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B', 'Majority class', 'Multinomial logistic', 'Random forest']
metrics = ['exact_outcome_accuracy', 'known_family_accuracy', 'unknown_cause_recall']
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
fig, axs = plt.subplots(1, 3, figsize=(13, 5.4), sharey=True)
for k, (ax, net) in enumerate(zip(axs, folders)):
    a = df[df.network.eq(net)].set_index('method').loc[methods, metrics].to_numpy() * 100
    assert a.shape == (8, 3) and np.isfinite(a).all()
    im = ax.imshow(a, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(8):
        for j in range(3):
            ax.text(j, i, f'{np.floor(a[i, j] * 10 + 0.50000001) / 10:.1f}', ha='center', va='center', fontsize=12, color='white' if a[i, j] >= 60 else '#222222')
    ax.set_xticks(range(3), ['Exact\noutcome\naccuracy', 'Known\nfamily\naccuracy', 'Unknown\ncause\nrecall'], fontsize=12)
    ax.set_yticks(range(8), methods, fontsize=12)
    ax.tick_params(length=0, pad=8)
    ax.axhline(4.5, color='#666666', linewidth=1)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{net} (n = {(18 if net == 'Anytown' else 21)})", fontsize=12, pad=12)
    ax.text(0.5, -0.24, f'({chr(97 + k)})', transform=ax.transAxes, ha='center', fontsize=12)
fig.subplots_adjust(left=0.18, right=0.91, bottom=0.23, top=0.9, wspace=0.12)
cax = fig.add_axes([0.93, 0.23, 0.015, 0.67])
fig.colorbar(im, cax=cax, label='Cases (%)')
out = base / '../figures/main/experiment2_three_networks_final'
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(out.with_suffix('.' + ext), dpi=300, bbox_inches='tight')
df.to_csv(out.with_suffix('.csv'), index=False)
