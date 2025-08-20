import csv
import os
import matplotlib.pyplot as plt
import numpy as np

output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "results.csv")

methods_order = [
    "naive CPU",
    "optimized CPU",
    "SpMV_OneThreadPerRow",
    "SpMV_OneWarpPerRow",
    "SpMV_coalescedBins",
    "SpMV_Hybrid"
]

colors = {
    "naive CPU": "tab:blue",
    "optimized CPU": "tab:orange",
    "SpMV_OneThreadPerRow": "tab:green",
    "SpMV_OneWarpPerRow": "tab:red",
    "SpMV_coalescedBins": "tab:purple",
    "SpMV_Hybrid": "tab:brown"
}

records = []
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['method'] not in methods_order:
            continue
        row['mean_time_ms'] = float(row['mean_time_ms'])
        row['bandwidth_gbps'] = float(row['bandwidth_gbps'])
        row['gflops'] = float(row['gflops'])
        records.append(row)

matrices_order = sorted(set(row['matrix'] for row in records))

bw_data = {}
time_data = {}
gflops_data = {}
for row in records:
    bw_data.setdefault(row['matrix'], {})[row['method']] = row['bandwidth_gbps']
    time_data.setdefault(row['matrix'], {})[row['method']] = row['mean_time_ms']
    gflops_data.setdefault(row['matrix'], {})[row['method']] = row['gflops']

def plot_bar(data_dict, ylabel, title, filename, normalize=False):
    matrices = [m for m in matrices_order if m in data_dict]
    num_methods = len(methods_order)
    num_matrices = len(matrices)
    group_width = 0.8
    bar_width = group_width / num_methods
    x = np.arange(num_matrices)

    if normalize:
        data_dict = {m: {k: v / max(vals.values()) for k, v in vals.items()} for m, vals in data_dict.items()}

    fig, ax = plt.subplots(figsize=(12,6))
    for i, method in enumerate(methods_order):
        offset = (i - (num_methods - 1) / 2) * bar_width
        values = [data_dict[m].get(method, 0) for m in matrices]
        ax.bar(x + offset, values, width=bar_width*0.9, label=method, color=colors.get(method, "tab:gray"))

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_bar(gflops_data, "GFLOPS", "GFLOPS by Method and Matrix", "gflops_plot.png")
plot_bar(bw_data, "Bandwidth (GB/s)", "Bandwidth by Method and Matrix", "bandwidth_plot.png")
plot_bar(time_data, "Normalized Time", "Normalized Mean Times by Method and Matrix", "normalized_time_plot.png", normalize=True)

print(f"Plots saved in {output_dir}: gflops_plot.png, bandwidth_plot.png, normalized_time_plot.png")
