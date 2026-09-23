import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(15, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 0.6])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Candidate and Job Data
candidates = [
    "Alex Chen\n(AI Eng)", "Sarah Miller\n(Data Analyst)", "David Clark\n(Jr Web)", "Marcus Vance\n(Chef)",
    "Elena Rostova\n(React Dev)", "Kevin Patel\n(Java Dev)", "Lisa Wong\n(UI/UX)", "Robert Taylor\n(Accountant)",
    "Tariq Mansoor\n(DevOps)", "Brian Adams\n(Sysadmin)", "Emily Watson\n(IT Support)", "James Sullivan\n(Civil PM)"
]

gemini_scores = [90.0, 55.0, 30.0, 5.0, 88.0, 48.0, 32.0, 5.0, 92.0, 52.0, 28.0, 5.0]
v1_0_scores = [96.3, 52.8, 45.9, 25.0, 94.1, 66.6, 55.7, 28.4, 90.8, 69.2, 35.7, 31.2]
v1_1_scores = [91.5, 35.4, 30.8, 15.0, 88.2, 53.0, 39.0, 15.0, 83.8, 56.1, 17.3, 15.0]
v1_4_scores = [91.8, 25.0, 30.8, 0.0, 86.9, 52.3, 38.9, 0.0, 84.8, 52.4, 17.0, 0.0]
v2_1_scores = [95.0, 29.3, 28.8, 0.0, 94.8, 45.2, 44.7, 0.0, 88.3, 55.3, 25.3, 0.0]
v2_2_scores = [94.1, 48.2, 25.6, 0.0, 95.7, 46.4, 44.0, 0.0, 86.3, 52.8, 16.9, 0.0]
v3_0_scores = [89.3, 48.2, 25.6, 0.0, 90.7, 45.2, 40.6, 0.0, 88.4, 51.1, 15.8, 0.0]

x = np.arange(len(candidates))
width = 0.35

# ----------------------------------------------------------------------
# PLOT 1: Trained System (v3.0 Production) vs. Gemini Ground Truth
# ----------------------------------------------------------------------
rects1 = ax1.bar(x - width/2, v3_0_scores, width, label='Your System v3.0 (Chunking + 143+ Skills + Date Parsing)', color='#1a5276', edgecolor='black', alpha=0.9)
rects2 = ax1.bar(x + width/2, gemini_scores, width, label='Gemini Ground Truth (Target Benchmark)', color='#e67e22', edgecolor='black', alpha=0.9)

ax1.set_title("System Match Scores vs. Gemini Ground Truth (v3.0 Engine: MAE 4.81%, Pearson r = 0.9900, 100% Accuracy)", fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel("Match Score (%)", fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(candidates, fontsize=8.5)
ax1.set_ylim(0, 115)
ax1.axhline(65, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='Highly Suitable Cutoff (>=65%)')
ax1.axhline(45, color='blue', linestyle='--', linewidth=1.2, alpha=0.7, label='Suitable Cutoff (45% - 64%)')
ax1.legend(loc='upper right', frameon=True, fontsize=9.5)

# Value annotations on bars
for rect in rects1:
    h = rect.get_height()
    if h > 0:
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax1.annotate('0%', xy=(rect.get_x() + rect.get_width() / 2, 0), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#b95d02')

# Draw job separators
ax1.axvline(3.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax1.axvline(7.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax1.text(1.5, 105, "JOB 1: Python & AI Engineer", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#2980b9"))
ax1.text(5.5, 105, "JOB 2: Frontend React Developer", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e67e22"))
ax1.text(9.5, 105, "JOB 3: DevOps & Cloud Engineer", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60"))

# ----------------------------------------------------------------------
# PLOT 2: Version Progression across Key Milestones
# ----------------------------------------------------------------------
w_prog = 0.17
rects_v10 = ax2.bar(x - 2*w_prog, v1_0_scores, w_prog, label='v1.0 (Baseline Cosine)', color='#e74c3c', alpha=0.7)
rects_v14 = ax2.bar(x - 1*w_prog, v1_4_scores, w_prog, label='v1.4 (Floor & Gatekeeper)', color='#f39c12', alpha=0.75)
rects_v21 = ax2.bar(x, v2_1_scores, w_prog, label='v2.1 (Skill Taxonomy Clusters)', color='#8e44ad', alpha=0.75)
rects_v22 = ax2.bar(x + 1*w_prog, v2_2_scores, w_prog, label='v2.2 (Transferability & Middle Fix)', color='#27ae60', alpha=0.9)
rects_gem = ax2.bar(x + 2*w_prog, gemini_scores, w_prog, label='Gemini Benchmark Target', color='#2980b9', alpha=0.85)

ax2.set_title("Full Version Progression: Evolution of Match Accuracy from v1.0 to v2.2", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Match Score (%)", fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(candidates, fontsize=8.5)
ax2.set_ylim(0, 115)
ax2.axvline(3.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax2.axvline(7.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

# ----------------------------------------------------------------------
# PLOT 3: Quantitative Metric Summary Table
# ----------------------------------------------------------------------
ax3.axis('off')
table_data = [
    ["System Version", "Classification Accuracy", "Top-1 Precision", "Mean Absolute Error (MAE)", "Pearson Correlation (r)", "Core Architectural Milestone"],
    ["v1.0 Baseline", "66.67% (8/12)", "100.0%", "14.04%", "0.9312", "Base SBERT + TF-IDF (Universal Cosine Floor Problem: Chef at 25%)"],
    ["v1.1 Rescaled", "91.67% (11/12)", "100.0%", "7.26%", "0.9620", "Linear Baseline Floor Subtraction (Shifted zero baseline by 0.22)"],
    ["v1.2 Scaled Bonus", "91.67% (11/12)", "100.0%", "5.90%", "0.9782", "Proportional Dynamic Experience Bonus (Chef & Accountant drop to 0.0%)"],
    ["v1.4 Gatekeeper", "91.67% (11/12)", "100.0%", "5.90%", "0.9782", "Technical Gatekeeper filter + Calibrated Cutoffs (65% / 40%)"],
    ["v2.0 Taxonomy", "83.33% (10/12)", "100.0%", "6.12%", "0.9805", "Skill Clusters & Taxonomy Affinity introduced (Discovered middle-point squeeze)"],
    ["v2.1 Sqrt Scaling", "91.67% (11/12)", "100.0%", "5.68%", "0.9821", "Square-root relevance scaling + Recalibrated thresholds (45% cutoff)"],
    ["v2.2 Transferability", "100.00% (12/12)", "100.0%", "5.76%", "0.9847", "Bounded Cluster Matching + Cross-Domain Career Transferability Matrix"],
    ["v3.0 Real Robust (Current)", "100.00% (12/12)", "100.0%", "4.81%", "0.9900", "Chunked Max-Pooling + 143+ Skills + Date Parser (Real Gap +38.68%, 0.00% Non-Tech FPR)"]
]

table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.2)
table.scale(1, 1.40)

# Style table header and highlighted final row
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 8:  # v3.0 row
        cell.set_facecolor('#d4efdf')
        cell.set_text_props(fontweight='bold', color='#145a32')
    elif row == 7:  # v2.2 row
        cell.set_facecolor('#ebf5fb')
        cell.set_text_props(fontweight='bold', color='#1b4f72')
    elif row % 2 == 1:
        cell.set_facecolor('#f8f9f9')

plt.tight_layout()

# Save images
out_local = "benchmark_comparison_graph.png"
plt.savefig(out_local, dpi=300, bbox_inches='tight')
print(f"Chart saved locally to {out_local}")

# Copy to artifact directory
artifact_dir = r"C:\Users\Kavinda\.gemini\antigravity\brain\bdd49f83-8001-4302-b026-c1256c609d6d"
if os.path.exists(artifact_dir):
    out_artifact = os.path.join(artifact_dir, "benchmark_comparison_graph.png")
    shutil.copyfile(out_local, out_artifact)
    print(f"Chart copied to artifact directory: {out_artifact}")
