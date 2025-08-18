import os
import re
import matplotlib.pyplot as plt
import numpy as np

input_dir = "./output"
img_dir = "./img"

os.makedirs(img_dir, exist_ok=True)

methods_order = [
    "naive CPU",
    "optimized CPU",
    "SpMV_OneThreadPerRow",
    "SpMV_OneWarpPerRow",
    "SpMV_coalescedBins",
    "SpMV_Hybrid"
]

patterns = {
    "naive CPU": re.compile(r"Using naive CPU SpMV"),
    "optimized CPU": re.compile(r"Using Optimized CPU SpMV with cache optimization"),
    "SpMV_OneThreadPerRow": re.compile(r"Using kernel:\s+SpMV_OneThreadPerRow"),
    "SpMV_OneWarpPerRow": re.compile(r"Using kernel:\s+SpMV_OneWarpPerRow"),
    "SpMV_coalescedBins": re.compile(r"Using kernel:\s+SpMV_coalescedBins"),
    "SpMV_Hybrid": re.compile(r"Using kernel:\s+SpMV_Hybrid"),
}

records = []

for filename in os.listdir(input_dir):
    if not filename.endswith(".out"):
        continue
    with open(os.path.join(input_dir, filename)) as f:
        content = f.read()

    method = None
    for m, patt in patterns.items():
        if patt.search(content):
            method = m
            break

    matrix_match = re.search(r"Using matrix:\s+\.\/mtx\/(.+\.mtx)", content)
    matrix = matrix_match.group(1) if matrix_match else "unknown"

    time_match = re.search(r"Mean time:\s+([\d\.]+) ms", content)
    mean_time = float(time_match.group(1)) if time_match else None

    bw_match = re.search(r"Bandwidth:\s+([\d\.]+) GB/s", content)
    bandwidth = float(bw_match.group(1)) if bw_match else None

    if None in (method, matrix, mean_time, bandwidth):
        print(f"Warning: incomplete data in {filename}")
        continue

    records.append({
        "filename": filename,
        "method": method,
        "matrix": matrix,
        "mean_time_ms": mean_time,
        "bandwidth_gbps": bandwidth
    })

# Process data directly for plotting

bw_data = {}
for row in records:
    bw_data.setdefault(row['matrix'], {})[row['method']] = row['bandwidth_gbps']

matrices = list(bw_data.keys())
num_methods = len(methods_order)
num_matrices = len(matrices)

group_width = 0.8
bar_width = group_width / num_methods
x = np.arange(num_matrices)

fig, ax = plt.subplots(figsize=(12,6))
for i, method in enumerate(methods_order):
    offset = (i - (num_methods - 1) / 2) * bar_width
    values = [bw_data[m].get(method, 0) for m in matrices]
    ax.bar(x + offset, values, width=bar_width * 0.9, label=method)
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=45, ha='right')
ax.set_ylabel("Bandwidth (GB/s)")
ax.set_title("Bandwidth by Method and Matrix")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "bandwidth_plot.png"))
plt.close()

time_data = {}
for row in records:
    time_data.setdefault(row['matrix'], {})[row['method']] = row['mean_time_ms']

normalized_time = {
    m: {k: v / max(times.values()) for k, v in times.items()}
    for m, times in time_data.items()
}

fig, ax = plt.subplots(figsize=(12,6))
for i, method in enumerate(methods_order):
    offset = (i - (num_methods - 1) / 2) * bar_width
    values = [normalized_time[m].get(method, 0) for m in matrices]
    ax.bar(x + offset, values, width=bar_width * 0.9, label=method)
ax.set_xticks(x)
ax.set_xticklabels(matrices, rotation=45, ha='right')
ax.set_ylabel("Normalized Time")
ax.set_title("Normalized Mean Times by Method and Matrix")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "normalized_time_plot.png"))
plt.close()

print("Plots saved: bandwidth_plot.png, normalized_time_plot.png")
