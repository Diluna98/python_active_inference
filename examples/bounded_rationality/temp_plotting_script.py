import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set clean styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# 1. Define and load the files
file_mapping = {
    "ploting_data_2.json": "2x2",
    "ploting_data_5.json": "5x5",
    "ploting_data_10.json": "10x10",
    "ploting_data_20.json": "20x20",
    "ploting_data_ours.json": "Ours (Dynamic)"
}

all_data = []

for filename, model_label in file_mapping.items():
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            raw_json = json.load(f)
            
            # Unroll lists into a long-form DataFrame structure
            latencies = raw_json["latency"]
            errors = raw_json["prediction_error"]
            
            for step, (lat, err) in enumerate(zip(latencies, errors)):
                all_data.append({
                    "Model": model_label,
                    "Step": step,
                    "Latency (ms)": lat,
                    "Prediction Error": err
                })
    else:
        print(f"Warning: {filename} not found, skipping.")

if not all_data:
    raise ValueError("No data files were successfully loaded. Check file paths!")

df = pd.DataFrame(all_data)

# Calculate summary averages for the trade-off plot
summary_df = df.groupby("Model").agg({
    "Latency (ms)": "mean",
    "Prediction Error": "mean"
}).reset_index()

# 2. Setup Plotting Canvas (Grid layout)
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# Color palette ensuring "Ours" stands out distinctively
colors = {"2x2": "#94a3b8", "5x5": "#64748b", "10x10": "#475569", "20x20": "#1e293b", "Ours (Dynamic)": "#dc2626"}

# --- Plot 1: The Trade-off Scatter ---
ax1 = fig.add_subplot(gs[0, 0])
sns.scatterplot(data=summary_df, x="Latency (ms)", y="Prediction Error", hue="Model", palette=colors, s=200, ax=ax1, zorder=3)
# Draw lines connecting baselines to visualize the Pareto Frontier
baselines = summary_df[summary_df["Model"] != "Ours (Dynamic)"].sort_values("Latency (ms)")
ax1.plot(baselines["Latency (ms)"], baselines["Prediction Error"], color='gray', linestyle='--', alpha=0.5, label="Baseline Frontier")

ax1.set_title("A: Efficiency vs. Accuracy Trade-off (Averages)")
ax1.set_xlabel("Mean Latency (ms) → [Lower is Better]")
ax1.set_ylabel("Mean Prediction Error → [Lower is Better]")

# --- Plot 2: Latency over Time/Steps ---
ax2 = fig.add_subplot(gs[0, 1])
sns.lineplot(data=df, x="Step", y="Latency (ms)", hue="Model", palette=colors, linewidth=2, ax=ax2)
ax2.set_title("B: Runtime Latency Dynamics Across Steps")
ax2.set_ylabel("Latency (ms)")

# --- Plot 3: Prediction Error over Time/Steps ---
ax3 = fig.add_subplot(gs[1, 0])
sns.lineplot(data=df, x="Step", y="Prediction Error", hue="Model", palette=colors, linewidth=2, ax=ax3)
ax3.set_title("C: Prediction Error Dynamics Across Steps")
ax3.set_ylabel("Prediction Error")

# --- Plot 4: Latency Distribution Profile ---
ax4 = fig.add_subplot(gs[1, 1])
sns.boxplot(data=df, x="Model", y="Latency (ms)", palette=colors, ax=ax4, width=0.5)
ax4.set_title("D: Latency Flexibility/Distribution Profile")
ax4.set_xlabel("Model Architecture")

# Final polishing
plt.suptitle("Active Inference Architecture Comparison: Fixed vs. Dynamic Multi-Resolution", fontsize=16, fontweight='bold', y=0.96)
plt.savefig("model_comparison_results.png", dpi=300, bbox_inches='tight')
plt.show()