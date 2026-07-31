import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

results_file = os.path.join(os.path.dirname(__file__), 'benchmark_results.txt')

exp1_data = []
exp2_data = []

with open(results_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_exp1 = False
in_exp2 = False

for line in lines:
    line = line.strip()
    if line == 'CSV_EXP1_START':
        in_exp1 = True
        continue
    elif line == 'CSV_EXP1_END':
        in_exp1 = False
        continue
    elif line == 'CSV_EXP2_START':
        in_exp2 = True
        continue
    elif line == 'CSV_EXP2_END':
        in_exp2 = False
        continue

    if in_exp1 and not line.startswith('kernel'):
        parts = line.split(',')
        if len(parts) == 6:
            exp1_data.append({
                'kernel': parts[0],
                'rows': int(parts[1]),
                'cols': int(parts[2]),
                'tree_us': float(parts[3]),
                'shuffle_us': float(parts[4]),
                'speedup': float(parts[5])
            })
    elif in_exp2 and not line.startswith('M'):
        parts = line.split(',')
        if len(parts) == 6:
            exp2_data.append({
                'M': int(parts[0]),
                'K': int(parts[1]),
                'N': int(parts[2]),
                'naive_us': float(parts[3]),
                'tiled_us': float(parts[4]),
                'speedup': float(parts[5])
            })

df1 = pd.DataFrame(exp1_data)
df2 = pd.DataFrame(exp2_data)

# ---------------------------------------------------------
# Plot 1: Exp 1 - LayerNorm & Softmax (Tree vs Warp Shuffle)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Filter for rows=1024 across different cols & config combinations
df1_ln = df1[df1['kernel'] == 'LayerNorm'].copy()
df1_sm = df1[df1['kernel'] == 'Softmax'].copy()

labels_ln = [f"{r}x{c}" for r, c in zip(df1_ln['rows'], df1_ln['cols'])]
x = np.arange(len(labels_ln))
width = 0.35

rects1 = ax1.bar(x - width/2, df1_ln['tree_us'], width, label='Shared Mem Tree', color='#e74c3c', alpha=0.85)
rects2 = ax1.bar(x + width/2, df1_ln['shuffle_us'], width, label='Warp Shuffle', color='#2ecc71', alpha=0.85)

ax1.set_ylabel('Tiempo de ejecución (μs)')
ax1.set_title('LayerNorm: Tree Reduction vs Warp Shuffle')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_ln, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Annotate speedup on top of bars
for i, (t, s, sp) in enumerate(zip(df1_ln['tree_us'], df1_ln['shuffle_us'], df1_ln['speedup'])):
    max_h = max(t, s)
    ax1.annotate(f'{sp:.2f}x',
                 xy=(x[i], max_h), xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color='#27ae60' if sp > 1.0 else '#c0392b')

labels_sm = [f"{r}x{c}" for r, c in zip(df1_sm['rows'], df1_sm['cols'])]
rects3 = ax2.bar(x - width/2, df1_sm['tree_us'], width, label='Shared Mem Tree', color='#e74c3c', alpha=0.85)
rects4 = ax2.bar(x + width/2, df1_sm['shuffle_us'], width, label='Warp Shuffle', color='#3498db', alpha=0.85)

ax2.set_ylabel('Tiempo de ejecución (μs)')
ax2.set_title('Softmax: Tree Reduction vs Warp Shuffle')
ax2.set_xticks(x)
ax2.set_xticklabels(labels_sm, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

for i, (t, s, sp) in enumerate(zip(df1_sm['tree_us'], df1_sm['shuffle_us'], df1_sm['speedup'])):
    max_h = max(t, s)
    ax2.annotate(f'{sp:.2f}x',
                 xy=(x[i], max_h), xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color='#2980b9' if sp > 1.0 else '#c0392b')

plt.tight_layout()
out_exp1_path = os.path.join(os.path.dirname(__file__), 'exp1_warp_shuffle.png')
plt.savefig(out_exp1_path, dpi=300)
plt.close()
print(f"Grafico Exp 1 guardado en: {out_exp1_path}")

# ---------------------------------------------------------
# Plot 2: Exp 2 - Linear GEMM Naive vs Tiling
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

labels_gemm = [f"{m}x{k}x{n}" for m, k, n in zip(df2['M'], df2['K'], df2['N'])]
x2 = np.arange(len(labels_gemm))

rects_g1 = ax1.bar(x2 - width/2, df2['naive_us'], width, label='Naive GEMM', color='#e67e22', alpha=0.85)
rects_g2 = ax1.bar(x2 + width/2, df2['tiled_us'], width, label='Shared Mem Tiling (16x16)', color='#9b59b6', alpha=0.85)

ax1.set_ylabel('Tiempo de ejecución (μs)')
ax1.set_yscale('log')
ax1.set_title('Tiempo de Ejecución: GEMM Naive vs Tiled (Escala Logarítmica)')
ax1.set_xticks(x2)
ax1.set_xticklabels(labels_gemm, rotation=45, ha='right', fontsize=8)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

for i, (t_n, t_t, sp) in enumerate(zip(df2['naive_us'], df2['tiled_us'], df2['speedup'])):
    max_h = max(t_n, t_t)
    ax1.annotate(f'{sp:.2f}x',
                 xy=(x2[i], max_h * 1.05), xytext=(0, 2),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color='#8e44ad')

# Speedup curve
ax2.plot(x2, df2['speedup'], marker='o', linewidth=2.5, color='#8e44ad', label='Speedup Factor')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax2.set_ylabel('Aceleración (Speedup X)')
ax2.set_title('Factor de Aceleración Obtenido por Shared Memory Tiling')
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_gemm, rotation=45, ha='right', fontsize=8)
ax2.set_ylim(0.7, max(df2['speedup']) * 1.15)
ax2.grid(True, linestyle='--', alpha=0.5)

for i, sp in enumerate(df2['speedup']):
    ax2.annotate(f'{sp:.2f}x',
                 xy=(x2[i], sp), xytext=(0, 6),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out_exp2_path = os.path.join(os.path.dirname(__file__), 'exp2_gemm_tiling.png')
plt.savefig(out_exp2_path, dpi=300)
plt.close()
print(f"Grafico Exp 2 guardado en: {out_exp2_path}")
