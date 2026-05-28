import re
import numpy as np
import pandas as pd

file_path = "log_5.txt"

values = []

with open(file_path, "r") as f:
    for line in f:
        if "context:" in line:
            match = re.search(r"context:\s*([0-9.]+)", line)
            if match:
                values.append(float(match.group(1)))

values = np.array(values)

bins = {
    "x < 220": [],
    "220 < x <= 222": [],
    "222 < x <= 225": [],
    "x > 225": []
}

for v in values:
    if v < 220:
        bins["x < 220"].append(v)
    elif 220 < v <= 222:
        bins["220 < x <= 222"].append(v)
    elif 222 < v <= 225:
        bins["222 < x <= 225"].append(v)
    else:
        bins["x > 225"].append(v)

results = []

for bin_name, arr in bins.items():
    arr = np.array(arr)
    if len(arr) == 0:
        mean = None
        std = None
        count = 0
    else:
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        count = len(arr)

    results.append({
        "bin": bin_name,
        "count": count,
        "mean": mean,
        "sigma": std
    })

df = pd.DataFrame(results)

print(df)