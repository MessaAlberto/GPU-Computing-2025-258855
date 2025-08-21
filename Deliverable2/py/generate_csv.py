import os
import re
import csv

input_dir = "./output"
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)
csv_file = os.path.join(results_dir, "results.csv")

records = []

# Regex patterns for methods
method_patterns = {
    "SpMV_OneThreadPerRow": re.compile(r"Using kernel:\s+SpMV_OneThreadPerRow"),
    "SpMV_OneWarpPerRow": re.compile(r"Using kernel:\s+SpMV_OneWarpPerRow"),
    "SpMV_coalescedBins": re.compile(r"Using kernel:\s+SpMV_coalescedBins"),
    "SpMV_Hybrid": re.compile(r"Using kernel:\s+SpMV_Hybrid"),
    "cuSPARSE SpMV": re.compile(r"Using kernel:\s+cuSPARSE SpMV")
}

for filename in os.listdir(input_dir):
    if not filename.endswith(".out"):
        continue

    path = os.path.join(input_dir, filename)
    with open(path) as f:
        content = f.read()

    # Determine method
    method = None
    for m, patt in method_patterns.items():
        if patt.search(content):
            method = m
            break

    # Extract matrix name
    matrix_match = re.search(r"Using matrix:\s+\.\/mtx\/(.+\.mtx)", content)
    matrix = matrix_match.group(1) if matrix_match else "unknown"

    # Extract block size
    block_match = re.search(r"Using block size:\s+(\d+)", content)
    block_size = int(block_match.group(1)) if block_match else None

    # Performance metrics
    time_match = re.search(r"Mean time:\s+([\d\.]+) ms", content)
    gflops_match = re.search(r"GFlops:\s+([\d\.]+)", content)
    bw_match = re.search(r"Bandwidth:\s+([\d\.]+) GB/s", content)

    record = {
        "filename": filename,
        "method": method,
        "matrix": matrix,
        "block_size": block_size,
        "mean_time_ms": float(time_match.group(1)) if time_match else None,
        "gflops": float(gflops_match.group(1)) if gflops_match else None,
        "bandwidth_gbps": float(bw_match.group(1)) if bw_match else None
    }

    records.append(record)

# Write CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

print(f"CSV saved to {csv_file}")
