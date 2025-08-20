import csv
import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

csv_d1 = "../Deliverable1/results/results.csv"
csv_d2 = "results/results.csv"

methods_order = [
    "SpMV_OneThreadPerRow",
    "SpMV_OneWarpPerRow",
    "SpMV_coalescedBins",
    "SpMV_Hybrid"
]

colors = {
    "SpMV_OneThreadPerRow": "tab:green",
    "SpMV_OneWarpPerRow": "tab:red",
    "SpMV_coalescedBins": "tab:purple",
    "SpMV_Hybrid": "tab:brown"
}

def load_csv(path):
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['method'] not in methods_order:
                continue
            row['mean_time_ms'] = float(row['mean_time_ms'])
            row['gflops'] = float(row['gflops'])
            row['bandwidth_gbps'] = float(row['bandwidth_gbps'])
            row['matrix'] = row['matrix']
            row['method'] = row['method']
            records.append(row)
    return records

data_d1 = load_csv(csv_d1)
data_d2 = load_csv(csv_d2)

data_d1_dict = {(r['matrix'], r['method']): r for r in data_d1}
data_d2_dict = {(r['matrix'], r['method']): r for r in data_d2}

improvements = {}
for key in data_d1_dict:
    if key not in data_d2_dict:
        continue
    matrix, method = key
    d1 = data_d1_dict[key]
    d2 = data_d2_dict[key]
    improvements.setdefault(matrix, {})[method] = {
        'time': (d1['mean_time_ms'] - d2['mean_time_ms']) / d1['mean_time_ms'] * 100 if d1['mean_time_ms']>0 else 0,
        'gflops': (d2['gflops'] - d1['gflops']) / d1['gflops'] * 100 if d1['gflops']>0 else 0,
        'bw': (d2['bandwidth_gbps'] - d1['bandwidth_gbps']) / d1['bandwidth_gbps'] * 100 if d1['bandwidth_gbps']>0 else 0
    }

matrices_order = sorted(improvements.keys())

def plot_bar_metric(metric_key, ylabel, title, filename):
    matrices = [m for m in matrices_order if m in improvements]
    num_methods = len(methods_order)
    num_matrices = len(matrices)
    x = np.arange(num_matrices)
    group_width = 0.8
    bar_width = group_width / num_methods

    fig, ax = plt.subplots(figsize=(12,6))
    for i, method in enumerate(methods_order):
        values = [improvements[m][method][metric_key] for m in matrices]
        offset = (i - (num_methods - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width*0.9, label=method, color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_bar_metric('time', 'Improvement (%)', 'Execution Time Improvement (D2 vs D1)', 'time_improvement.png')
plot_bar_metric('gflops', 'Improvement (%)', 'GFLOPS Improvement (D2 vs D1)', 'gflops_improvement.png')
plot_bar_metric('bw', 'Improvement (%)', 'Bandwidth Improvement (D2 vs D1)', 'bandwidth_improvement.png')

print("Images saved in results/: time_improvement.png, gflops_improvement.png, bandwidth_improvement.png")
