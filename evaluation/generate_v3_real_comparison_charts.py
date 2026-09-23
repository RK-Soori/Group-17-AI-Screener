import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.1, 0.6])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Data for 15 Real Categories (v2.2 vs v3.0)
categories = [
    "Python Dev\n(Direct)", "Data Science\n(Direct)", "React Dev\n(Direct)", "Web Design\n(Direct)", "DevOps\n(Direct)",
    "DotNet Dev\n(Adjacent)", "Java Dev\n(Adjacent)", "Database\n(Adjacent)", "NetSec Eng\n(Adjacent)",
    "Accountant\n(Irrelevant)", "Civil Eng\n(Irrelevant)", "Sales\n(Irrelevant)", "Food & Bev\n(Irrelevant)", "HR\n(Irrelevant)", "Arts\n(Irrelevant)"
]

v2_scores = [29.89, 26.83, 56.99, 28.36, 47.99, 35.14, 27.38, 5.99, 14.15, 2.00, 1.67, 0.60, 0.00, 0.00, 0.00]
v3_scores = [32.46, 32.48, 57.34, 37.33, 54.08, 38.21, 30.50, 11.73, 16.78, 2.00, 1.46, 0.60, 0.00, 0.00, 0.00]

x = np.arange(len(categories))
width = 0.38

# PLOT 1: Real-World Candidate Category Comparison (v2.2 vs v3.0)
rects1 = ax1.bar(x - width/2, v2_scores, width, label='Engine v2.2 (Base 50 Skills, Truncated SBERT)', color='#7f8c8d', edgecolor='black', alpha=0.85)
rects2 = ax1.bar(x + width/2, v3_scores, width, label='Engine v3.0 (143+ Skills, Chunked Max-Pooling, Date Parser)', color='#2980b9', edgecolor='black', alpha=0.9)

ax1.set_title("Real-World Resume Dataset: Performance Comparison by Category (v2.2 vs. v3.0 Robustness Engine)", fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel("Average Match Score (%)", fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=8.5)
ax1.set_ylim(0, 75)
ax1.axhline(45, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='Suitable Domain Threshold (>=45%)')
ax1.legend(loc='upper right', frameon=True, fontsize=10)

# Vertical category separators
ax1.axvline(4.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax1.axvline(8.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax1.text(2.0, 68, "DIRECT TARGET PROFESSIONS", ha='center', fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#d4efdf", ec="#27ae60"))
ax1.text(6.5, 68, "ADJACENT CROSS-DOMAIN", ha='center', fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#2980b9"))
ax1.text(11.5, 68, "IRRELEVANT NON-TECH", ha='center', fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c"))

# Bar labels
for rect in rects1:
    h = rect.get_height()
    if h > 0:
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='#424949')

for rect in rects2:
    h = rect.get_height()
    if h > 0:
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#154360')

# PLOT 2: Aggregate Tier Comparison & Discrimination Gap
tiers = ["Target Technical Domain\n(Direct Match)", "Adjacent Cross-Domain\n(Transferable)", "Irrelevant Non-Technical\n(Non-Tech Candidates)"]
v2_tier_means = [35.26, 24.30, 0.71]
v3_tier_means = [39.71, 27.69, 0.68]
v2_tier_hit_rates = [26.7, 11.7, 0.0]
v3_tier_hit_rates = [35.0, 11.7, 0.0]

x_tier = np.arange(len(tiers))
width_tier = 0.30

rects_t1 = ax2.bar(x_tier - width_tier/2, v2_tier_means, width_tier, label='v2.2 Mean Score', color='#95a5a6', edgecolor='black', alpha=0.85)
rects_t2 = ax2.bar(x_tier + width_tier/2, v3_tier_means, width_tier, label='v3.0 Mean Score', color='#27ae60', edgecolor='black', alpha=0.9)

ax2.set_title("Candidate Tier Aggregate Separation & Statistical Discrimination Gap", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Mean Match Score (%)", fontsize=11, fontweight='bold')
ax2.set_xticks(x_tier)
ax2.set_xticklabels(tiers, fontsize=10, fontweight='bold')
ax2.set_ylim(0, 55)
ax2.legend(loc='upper right', frameon=True, fontsize=10)

for rect in rects_t1:
    h = rect.get_height()
    ax2.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#515a5a')

for rect in rects_t2:
    h = rect.get_height()
    ax2.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#145a32')

# Gap annotation
ax2.annotate(f"v2.2 Separation: +34.55%", xy=(0.85, 30), fontsize=10, fontweight='bold', color='#7f8c8d',
             bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="#7f8c8d"))
ax2.annotate(f"v3.0 Separation: +39.03% (+4.48% wider gap!)", xy=(0.85, 23), fontsize=10, fontweight='bold', color='#145a32',
             bbox=dict(boxstyle="square,pad=0.4", fc="#d4efdf", ec="#27ae60"))

# PLOT 3: Table Summary
ax3.axis('off')
table_data = [
    ["Metric / Characteristic", "Engine v2.2 (Previous)", "Engine v3.0 (Robustness Release)", "Empirical Delta / Benefit"],
    ["Technical Skill Vocabulary", "50 hardcoded terms", "143+ industry terms across 4 clusters", "+93 technical competencies recognized"],
    ["Experience Extraction Logic", "Simple regex ((\d+) years)", "Explicit tenure + Multi-span dates (YYYY-YYYY)", "Recovers 4-10 yrs missing tenure on messy CVs"],
    ["Document Length Ingestion", "256 tokens max (~200 words)", "Sliding window (140 words, 35 overlap, max-pool)", "Full 100% visibility of 2-page multi-project CVs"],
    ["Target Domain Mean Score", "35.26%", "39.71%", "+4.45% boost on qualified real resumes"],
    ["Target Hit Rate (>=45%)", "26.70%", "35.00%", "+8.30% increase in qualified shortlist rate"],
    ["Non-Technical FPR (>=45%)", "0.00% (0 / 60 resumes)", "0.00% (0 / 60 resumes)", "Strictly 0.00% False Positives preserved"],
    ["Non-Technical Hard Zero Rate", "93.33%", "93.33%", "Irrelevant applicants firmly rejected at 0.0%"],
    ["Gemini Benchmark MAE", "5.76%", "4.81%", "-0.95% error reduction against Gemini"],
    ["Gemini Benchmark Pearson r", "0.9847", "0.9900", "+0.0053 stronger correlation with Gemini"]
]

table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.2)
table.scale(1, 1.35)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 1:
        cell.set_facecolor('#f8f9f9')
    if col == 3:
        cell.set_text_props(fontweight='bold', color='#1e8449')

plt.tight_layout()

out_local = "v3_real_benchmark_comparison.png"
plt.savefig(out_local, dpi=300, bbox_inches='tight')
print(f"Chart saved locally to {out_local}")

artifact_dir = r"C:\Users\Kavinda\.gemini\antigravity\brain\bdd49f83-8001-4302-b026-c1256c609d6d"
if os.path.exists(artifact_dir):
    out_artifact = os.path.join(artifact_dir, "v3_real_benchmark_comparison.png")
    shutil.copyfile(out_local, out_artifact)
    print(f"Chart copied to artifact directory: {out_artifact}")
