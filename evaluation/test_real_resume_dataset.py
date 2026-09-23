import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add web_app to path
sys.path.insert(0, os.path.abspath('web_app'))
from nlp_engine import calculate_match_scores

print("=" * 80)
print("EVALUATING PRODUCTION AI ENGINE (v2.2) ON REAL-WORLD RESUME DATASET (13,389 RESUMES)")
print("=" * 80)

# Load real dataset
csv_path = 'Resume_Dataset_Real.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

print(f"Loading real resumes from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} total real resumes across {df['Category'].nunique()} categories.")

# Define 3 Industry Benchmark Job Descriptions
JOBS = [
    {
        "job_id": "REAL-JOB-1",
        "role": "Senior Python & AI Engineer",
        "description": "Looking for a Senior Python & AI Engineer with at least 3 years of experience. Must be proficient in Python, PyTorch or TensorFlow, Natural Language Processing, building REST APIs with FastAPI or Flask, and containerization using Docker and Linux.",
        "test_categories": {
            "Target (High Match)": ["Python Developer", "Data Science"],
            "Adjacent (Mid Match)": ["Java Developer", "Database"],
            "Irrelevant (Zero Match)": ["Accountant", "Food and Beverages"]
        }
    },
    {
        "job_id": "REAL-JOB-2",
        "role": "Frontend React Developer",
        "description": "Hiring a Frontend Developer with 2+ years experience building responsive web apps using React, JavaScript, TypeScript, HTML5, CSS3, and Redux. Experience with Git and REST APIs required.",
        "test_categories": {
            "Target (High Match)": ["React Developer", "Web Designing"],
            "Adjacent (Mid Match)": ["Java Developer", "DotNet Developer"],
            "Irrelevant (Zero Match)": ["Civil Engineer", "Sales"]
        }
    },
    {
        "job_id": "REAL-JOB-3",
        "role": "DevOps & Cloud Engineer",
        "description": "Seeking a DevOps & Cloud Engineer with 3+ years experience in AWS, Docker, Kubernetes, CI/CD pipelines (Jenkins/GitHub Actions), Terraform, and Linux administration.",
        "test_categories": {
            "Target (High Match)": ["DevOps", "Network Security Engineer"],
            "Adjacent (Mid Match)": ["Python Developer", "Java Developer"],
            "Irrelevant (Zero Match)": ["Human Resources", "Arts"]
        }
    }
]

# Evaluation loop
SAMPLE_SIZE_PER_CAT = 10
all_evaluations = []

for job in JOBS:
    job_role = job["role"]
    job_desc = job["description"]
    print(f"\nEvaluating: {job_role}...")
    
    for tier_name, cat_list in job["test_categories"].items():
        for cat in cat_list:
            subset = df[df['Category'] == cat]
            if len(subset) == 0:
                print(f"Warning: Category '{cat}' has 0 records.")
                continue
            
            # Sample deterministically for reproducibility
            samples = subset.head(SAMPLE_SIZE_PER_CAT)
            
            for idx, row in samples.iterrows():
                resume_text = str(row['Text'])
                # Run production engine
                res = calculate_match_scores(job_desc, [resume_text])
                score = float(res[0]['score'])
                label = res[0]['label']
                
                all_evaluations.append({
                    "job_role": job_role,
                    "tier": tier_name,
                    "candidate_category": cat,
                    "score": score,
                    "label": label,
                    "resume_char_len": len(resume_text)
                })

eval_df = pd.DataFrame(all_evaluations)

print("\n" + "=" * 80)
print("REAL RESUME BENCHMARK RESULTS SUMMARY")
print("=" * 80)

summary_by_tier = eval_df.groupby(['job_role', 'tier'])['score'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
print("\nScore Statistics by Job Role & Candidate Tier:")
print(summary_by_tier.to_string(index=False))

overall_by_tier = eval_df.groupby('tier')['score'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
print("\nOverall Aggregate Score by Candidate Tier:")
print(overall_by_tier.to_string(index=False))

# Label distribution across tiers
label_dist = pd.crosstab(eval_df['tier'], eval_df['label'], normalize='index') * 100
print("\nClassification Label Distribution (% by Tier):")
print(label_dist.round(2).to_string())

# Key Performance Indicators
target_df = eval_df[eval_df['tier'] == 'Target (High Match)']
adjacent_df = eval_df[eval_df['tier'] == 'Adjacent (Mid Match)']
irrelevant_df = eval_df[eval_df['tier'] == 'Irrelevant (Zero Match)']

target_hit_rate = (target_df['label'].isin(['Highly Suitable', 'Suitable'])).mean() * 100
false_positive_rate = (irrelevant_df['score'] >= 45.0).mean() * 100
non_tech_zero_rate = (irrelevant_df['score'] == 0.0).mean() * 100
discrimination_gap = target_df['score'].mean() - irrelevant_df['score'].mean()

print("\n" + "-" * 50)
print(f"Target Domain Hit Rate (Suitable/High): {target_hit_rate:.2f}%")
print(f"Non-Technical False Positive Rate (>=45%): {false_positive_rate:.2f}%")
print(f"Non-Technical Hard Zero Rate (==0.0%): {non_tech_zero_rate:.2f}%")
print(f"Discrimination Gap (Target Mean - Irrelevant Mean): +{discrimination_gap:.2f}%")
print("-" * 50)

# Save JSON results
output_json = {
    "total_real_resumes_tested": len(eval_df),
    "sample_categories": list(eval_df['candidate_category'].unique()),
    "kpis": {
        "target_hit_rate_percent": round(target_hit_rate, 2),
        "false_positive_rate_percent": round(false_positive_rate, 2),
        "non_technical_zero_rate_percent": round(non_tech_zero_rate, 2),
        "discrimination_gap_percent": round(discrimination_gap, 2)
    },
    "tier_aggregates": overall_by_tier.to_dict(orient='records'),
    "category_breakdown": eval_df.groupby('candidate_category')['score'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index().to_dict(orient='records')
}

with open("real_resume_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=4)
print("\nDetailed metrics saved to 'real_resume_benchmark_results.json'.")

# ----------------------------------------------------------------------
# GENERATE HIGH-RESOLUTION COMPARISON CHART ON REAL RESUMES
# ----------------------------------------------------------------------
print("\nGenerating visual chart 'real_resume_benchmark_graph.png'...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Mean Match Score by Profession Category
cat_order = [
    # Target
    "Python Developer", "Data Science", "React Developer", "DevOps",
    # Adjacent
    "Network Security Engineer", "Web Designing", "Java Developer", "DotNet Developer", "Database",
    # Irrelevant
    "Accountant", "Food and Beverages", "Civil Engineer", "Sales", "Human Resources", "Arts"
]
cat_means = eval_df.groupby('candidate_category')['score'].mean().reindex(cat_order).dropna()

colors = []
for c in cat_means.index:
    if c in ["Python Developer", "Data Science", "React Developer", "DevOps"]:
        colors.append('#27ae60') # Green
    elif c in ["Network Security Engineer", "Web Designing", "Java Developer", "DotNet Developer", "Database"]:
        colors.append('#2980b9') # Blue
    else:
        colors.append('#c0392b') # Red

bars = ax1.barh(cat_means.index, cat_means.values, color=colors, edgecolor='black', alpha=0.85)
ax1.set_xlim(0, 100)
ax1.set_xlabel("Average Match Score (%)", fontsize=11, fontweight='bold')
ax1.set_title("Match Scores Across Real-World Candidate Professions", fontsize=12, fontweight='bold', pad=12)
ax1.axvline(65, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='Highly Suitable (>=65%)')
ax1.axvline(45, color='blue', linestyle='--', linewidth=1.2, alpha=0.7, label='Suitable (45%-64%)')

for bar in bars:
    w = bar.get_width()
    ax1.annotate(f'{w:.1f}%', xy=(w, bar.get_y() + bar.get_height()/2),
                 xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')

ax1.legend(loc='lower right', frameon=True, fontsize=10)

# Plot 2: Score Distribution by Candidate Tier (Boxplot / Scatter)
tier_order = ['Target (High Match)', 'Adjacent (Mid Match)', 'Irrelevant (Zero Match)']
tier_data = [eval_df[eval_df['tier'] == t]['score'].values for t in tier_order]

bp = ax2.boxplot(tier_data, tick_labels=['Target Roles\n(Direct Match)', 'Adjacent Roles\n(Cross-Domain)', 'Irrelevant Roles\n(Non-Technical)'],
                 patch_artist=True, widths=0.5, medianprops=dict(color='black', linewidth=2))

tier_colors = ['#2ecc71', '#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], tier_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add jittered points to show all real candidates
for i, scores in enumerate(tier_data):
    jitter = np.random.normal(i + 1, 0.05, size=len(scores))
    ax2.scatter(jitter, scores, alpha=0.5, color='black', s=18, edgecolors='none')

ax2.set_ylim(-5, 105)
ax2.set_ylabel("Match Score (%)", fontsize=11, fontweight='bold')
ax2.set_title("Distribution & Clear Separation Across Candidate Tiers", fontsize=12, fontweight='bold', pad=12)
ax2.axhline(65, color='green', linestyle='--', linewidth=1.2, alpha=0.7)
ax2.axhline(45, color='blue', linestyle='--', linewidth=1.2, alpha=0.7)

# Add annotation boxes
ax2.text(1, 95, f"Mean: {eval_df[eval_df['tier']=='Target (High Match)']['score'].mean():.1f}%", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", fc="#d4efdf", ec="green"))
ax2.text(2, 60, f"Mean: {eval_df[eval_df['tier']=='Adjacent (Mid Match)']['score'].mean():.1f}%", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", fc="#d6eaf8", ec="blue"))
ax2.text(3, 20, f"Mean: {eval_df[eval_df['tier']=='Irrelevant (Zero Match)']['score'].mean():.1f}%\nZeroed: {non_tech_zero_rate:.0f}%", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle="round", fc="#fadbd8", ec="red"))

plt.tight_layout()
out_img = "real_resume_benchmark_graph.png"
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Chart saved locally to {out_img}")

# Copy to artifact directory
artifact_dir = r"C:\Users\Kavinda\.gemini\antigravity\brain\bdd49f83-8001-4302-b026-c1256c609d6d"
if os.path.exists(artifact_dir):
    import shutil
    out_artifact = os.path.join(artifact_dir, out_img)
    shutil.copyfile(out_img, out_artifact)
    print(f"Chart copied to artifact directory: {out_artifact}")

print("\nREAL RESUME BENCHMARK EVALUATION COMPLETED SUCCESSFULLY!")
