import os
import csv
import matplotlib.pyplot as plt

output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, "results.csv")

# Matrix to analyze
matrix_name = "ecology1"

methods_order = [
    "SpMV_OneThreadPerRow",
    "SpMV_OneWarpPerRow",
    "SpMV_coalescedBins",
    "SpMV_Hybrid"
]

records = []

with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Accept both ecology1 or ecology1.mtx depending on CSV
        if row["matrix"].replace(".mtx", "") == matrix_name:
            row["block_size"] = int(row["block_size"]) if row["block_size"] else None
            row["bandwidth_gbps"] = float(row["bandwidth_gbps"])
            records.append(row)

block_sizes = sorted({r["block_size"] for r in records if r["block_size"] is not None})

plt.figure(figsize=(10, 6))

for method in methods_order:
    method_records = [r for r in records if r["method"] == method]
    if not method_records:
        continue
    y = []
    for bs in block_sizes:
        vals = [r["bandwidth_gbps"] for r in method_records if r["block_size"] == bs]
        y.append(sum(vals) / len(vals) if vals else 0)
    plt.plot(block_sizes, y, marker='o', label=method)

plt.xlabel("Block Size")
plt.ylabel("Bandwidth (GB/s)")
plt.title(f"Bandwidth vs Block Size for Different Kernels on {matrix_name}")
plt.xticks(block_sizes)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save image with name like ecology1_blocksize.png
img_path = os.path.join(output_dir, f"{matrix_name}_blocksize.png")
plt.savefig(img_path)
plt.show()

print(f"Plot saved as {img_path}")
