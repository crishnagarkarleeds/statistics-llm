import matplotlib.pyplot as plt
import numpy as np
from math import pi

models = ['DeepSeek', 'LLaMA', 'LLaMA FT', 'Mistral', 'Mistral FT']
dimensions = ['Correctness', 'Explanation', 'Reasoning', 'Weighted']

evaluation_data = {
    'Correctness': {
        'Perplexity': [0.040, -0.089, 0.029, 0.101, -0.126],
        'BLEU':       [0.020, 0.208, 0.195, 0.186, -0.016],
        'SBERT Sim':  [0.002, 0.183, 0.139, 0.146, -0.020],
        'BERTScore':  [0.207, 0.338, 0.089, 0.039, -0.050],
        'DeepSeek Judge': [0.402, -0.050, 0.293, 0.030, 0.287],
        'LLaMA Judge':    [0.378, 0.308, 0.541, 0.587, 0.576],
        'Mistral Judge':  [0.476, 0.319, 0.538, 0.489, 0.603],
    },
    'Explanation': {
        'Perplexity': [-0.080, -0.047, 0.022, 0.097, -0.101],
        'BLEU':       [0.143, 0.201, 0.120, 0.165, -0.074],
        'SBERT Sim':  [0.084, 0.135, 0.032, 0.124, -0.038],
        'BERTScore':  [0.202, 0.138, -0.010, 0.018, -0.040],
        'DeepSeek Judge': [0.293, 0.273, 0.265, 0.024, 0.265],
        'LLaMA Judge':    [0.359, 0.180, 0.412, 0.651, 0.453],
        'Mistral Judge':  [0.229, 0.224, 0.414, 0.492, 0.639],
    },
    'Reasoning': {
        'Perplexity': [-0.097, -0.053, 0.078, 0.026, -0.118],
        'BLEU':       [0.046, 0.233, 0.144, 0.155, -0.037],
        'SBERT Sim':  [0.090, 0.236, 0.108, 0.131, -0.008],
        'BERTScore':  [0.097, 0.236, 0.017, 0.066, -0.008],
        'DeepSeek Judge': [0.216, 0.105, 0.208, 0.004, 0.138],
        'LLaMA Judge':    [0.313, 0.295, 0.484, 0.673, 0.536],
        'Mistral Judge':  [0.251, 0.303, 0.437, 0.512, 0.496],
    },
    'Weighted': {
        'Perplexity': [-0.037, -0.080, 0.056, 0.083, -0.125],
        'BLEU':       [0.084, 0.224, 0.153, 0.186, -0.039],
        'SBERT Sim':  [0.039, 0.207, 0.103, 0.143, -0.026],
        'BERTScore':  [0.200, 0.252, 0.042, 0.045, -0.052],
        'DeepSeek Judge': [0.370, 0.180, 0.306, 0.017, 0.311],
        'LLaMA Judge':    [0.424, 0.293, 0.525, 0.654, 0.567],
        'Mistral Judge':  [0.411, 0.304, 0.530, 0.546, 0.629],
    }
}

# --- Geometry ---
N = len(models)
angles = [n/float(N)*2*pi for n in range(N)]
angles += angles[:1]   # close with the starting angle

# --- Compute offset to avoid negative r ---
min_val = min(min(min(vs) for vs in dim.values()) for dim in evaluation_data.values())
max_val = max(max(max(vs) for vs in dim.values()) for dim in evaluation_data.values())
offset = -min_val if min_val < 0 else 0.0       # ~ 0.126
r_max  = max_val + offset                        # top of scale
zero_r = offset                                  # radius where τ=0 lies

# --- Styles ---
styles_trad = {
    'BLEU':      dict(color='#B0B0B0', ls='-',  lw=1.8, marker='o', alpha=0.5),
    'BERTScore': dict(color='#A0A0A0', ls='--', lw=1.8, marker='s', alpha=0.5),
    'SBERT':     dict(color='#C0C0C0', ls='-.', lw=1.8, marker='^', alpha=0.5),
    'Perplexity':dict(color='#909090', ls=':',  lw=1.8, marker='v', alpha=0.5),
}
styles_judge = {
    'DeepSeek Judge': dict(color='#E31A1C', ls='-', lw=3.4, marker='o'),
    'LLaMA Judge':    dict(color='#1F78B4', ls='-',  lw=3.8, marker='s'),
    'Mistral Judge':  dict(color='#33A02C', ls='-.', lw=3.4, marker='^'),
}

fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), subplot_kw=dict(projection='polar'))
plt.subplots_adjust(left=0.18, right=0.98, top=0.87, bottom=0.12, wspace=0.35)

for i, dim in enumerate(dimensions):
    ax = axes[i]
    ax.set_title(dim, fontsize=18, pad=20)

    # Orientation (keep charts consistent)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Scale & faint rings (no numbers)
    ax.set_ylim(0, r_max)
    rings = np.linspace(0, r_max, 8)[1:]   # skip center
    ax.set_yticks(rings); ax.set_yticklabels([])
    ax.grid(True, alpha=0.12, linewidth=0.6)
    ax.spines['polar'].set_alpha(0.15)

    # Zero reference ring (τ=0) - more prominent
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(th, np.full_like(th, zero_r), color='#444', lw=1.5, alpha=0.7, zorder=2)
    
    # Add τ=0 label for the first subplot only
    if i == 0:
        ax.text(0, zero_r * 1.05, 'τ=0', fontsize=10, ha='center', va='bottom', 
                color='#444', alpha=0.8, weight='bold')

    # Axis labels with better positioning
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models, fontsize=13)
    # Push labels slightly outward to avoid overlap
    ax.tick_params(axis='x', pad=8)

    # Plot series with proper negative handling
    for name, sty in {**styles_trad, **styles_judge}.items():
        vals = evaluation_data[dim][name if name != 'SBERT' else 'SBERT Sim']  # Map SBERT back to data key
        vals_shift = [v + offset for v in vals] + [vals[0] + offset]  # Apply offset to handle negatives
        ax.plot(angles, vals_shift, markersize=7, zorder=3, **sty)

# Legends - FIXED VERSION
from matplotlib.lines import Line2D
trad_handles = [Line2D([0],[0], **{k:styles_trad['BLEU'][k] for k in ('color','ls','lw','alpha')}, marker='o', label='BLEU'),
                Line2D([0],[0], **{k:styles_trad['BERTScore'][k] for k in ('color','ls','lw','alpha')}, marker='s', label='BERTScore'),
                Line2D([0],[0], **{k:styles_trad['SBERT'][k] for k in ('color','ls','lw','alpha')}, marker='^', label='SBERT'),
                Line2D([0],[0], **{k:styles_trad['Perplexity'][k] for k in ('color','ls','lw','alpha')}, marker='v', label='Perplexity')]

judge_handles = [Line2D([0],[0], **{k:styles_judge['DeepSeek Judge'][k] for k in ('color','ls','lw')}, marker='o', label='DeepSeek Judge'),
                 Line2D([0],[0], **{k:styles_judge['LLaMA Judge'][k] for k in ('color','ls','lw')}, marker='s', label='LLaMA Judge'),
                 Line2D([0],[0], **{k:styles_judge['Mistral Judge'][k] for k in ('color','ls','lw')}, marker='^', label='Mistral Judge')]

# Extract labels from handles
trad_labels = [handle.get_label() for handle in trad_handles]
judge_labels = [handle.get_label() for handle in judge_handles]

# Create legends with proper syntax
fig.legend(trad_handles, trad_labels, title='Traditional Metrics', loc='upper left',
           bbox_to_anchor=(0.02, 0.98), fontsize=12, title_fontsize=13, frameon=True, fancybox=True)
fig.legend(judge_handles, judge_labels, title='LLM-as-Judge', loc='upper left',
           bbox_to_anchor=(0.02, 0.68), fontsize=12, title_fontsize=13, frameon=True, fancybox=True)

plt.savefig('table4_radar_chart_shifted.pdf', dpi=300, bbox_inches='tight')
plt.savefig('table4_radar_chart_shifted.png', dpi=300, bbox_inches='tight')
plt.show()
