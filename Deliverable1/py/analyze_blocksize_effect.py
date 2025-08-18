import os
import re
import matplotlib.pyplot as plt
import numpy as np

input_dir = "./output"
img_dir = "./img"
os.makedirs(img_dir, exist_ok=True)
matrix_name = "ecology1"

methods_order = [
    "SpMV_OneThreadPerRow",
    "SpMV_OneWarpPerRow",
    "SpMV_coalescedBins",
    "SpMV_Hybrid"
]

patterns = {
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
    if matrix != f"{matrix_name}.mtx":
        continue

    block_size_match = re.search(r"Using block size:\s+(\d+)", content)
    block_size = int(block_size_match.group(1)) if block_size_match else None

    bw_match = re.search(r"Bandwidth:\s+([\d\.]+) GB/s", content)
    bandwidth = float(bw_match.group(1)) if bw_match else None

    if None in (method, block_size, bandwidth):
        print(f"Warning: incomplete data in {filename}")
        continue

    records.append({
        "method": method,
        "block_size": block_size,
        "bandwidth_gbps": bandwidth
    })

data = {}
for r in records:
    data.setdefault(r['method'], {})[r['block_size']] = r['bandwidth_gbps']

block_sizes = sorted(set(r['block_size'] for r in records))

plt.figure(figsize=(10,6))
for method in methods_order:
    y = [data.get(method, {}).get(bs, 0) for bs in block_sizes]
    plt.plot(block_sizes, y, marker='o', label=method)

plt.xlabel("Block Size")
plt.ylabel("Bandwidth (GB/s)")
plt.title(f"Bandwidth vs Block Size for Different Kernels on {matrix_name}")
plt.xticks(block_sizes)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(img_dir, f"{matrix_name}_blocksize.png"))
plt.show()
print(f"Plot saved as {matrix_name}_blocksize.png")
