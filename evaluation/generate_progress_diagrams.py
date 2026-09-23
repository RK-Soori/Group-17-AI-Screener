import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# Generate High-Resolution Architecture & Workflow Diagrams for Stage 2
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
ax.axis('off')

# Title
ax.text(0.5, 0.96, "AI-Based Candidate Screening System: End-to-End System Architecture (v3.0)",
        fontsize=16, fontweight='bold', ha='center', va='top', color='#1a365d')
ax.text(0.5, 0.93, "Group 17 - Essentials of AI (Stage 2 Progress Review)",
        fontsize=12, style='italic', ha='center', va='top', color='#4a5568')

def draw_box(ax, x, y, w, h, title, subtitle, color, border_color='#2b6cb0'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1a202c')
    ax.text(x + w/2, y + h*0.35, subtitle, ha='center', va='center', fontsize=8.2, color='#4a5568', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.8, shrinkA=3, shrinkB=3))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, label, ha='center', va='bottom', fontsize=8, color='#4a5568', fontweight='bold')

# Layer 1: Input Layer
draw_box(ax, 0.05, 0.72, 0.25, 0.14, "1. Job Description Input", "Title, Core Skills, Min. Exp, Role Duties\nRaw Text or Structured Prompt", "#ebf8ff", "#3182ce")
draw_box(ax, 0.05, 0.52, 0.25, 0.14, "2. Candidate CV Input", "PDF/DOCX Extractor, Multi-page Text\n13,389 Real-World Resumes Corpus", "#ebf8ff", "#3182ce")

# Layer 2: Preprocessing & Feature Extraction
draw_box(ax, 0.36, 0.76, 0.26, 0.12, "Pre-processing & Tokenization", "HR Stopwords Filter, Synonyms Normalizer\nKeyword Frequency Reinforcement", "#feebc8", "#dd6b20")
draw_box(ax, 0.36, 0.60, 0.26, 0.12, "Multi-Span Tenure Parser", "Explicit ('5+ yrs') + Multi-Interval Dates\n(YYYY-YYYY, YYYY-Present)", "#feebc8", "#dd6b20")
draw_box(ax, 0.36, 0.44, 0.26, 0.12, "Sliding-Window Chunker", "140 Words/Window, 35 Words Overlap\nOvercomes 256-Token Truncation", "#feebc8", "#dd6b20")

# Layer 3: Hybrid AI Engine
draw_box(ax, 0.68, 0.78, 0.27, 0.10, "A. Domain SBERT (45%)", "Fine-Tuned MiniLM-L6 (RTX 4050 GPU)\nMax-Pooled Chunk Cosine Sim", "#e6fffa", "#319795")
draw_box(ax, 0.68, 0.65, 0.27, 0.10, "B. Domain TF-IDF (35%)", "2,076 Docs, 48,393 N-Grams\nSublinear Frequency Term Cosine", "#e6fffa", "#319795")
draw_box(ax, 0.68, 0.52, 0.27, 0.10, "C. Bounded Skill Taxonomy (15%)", "143+ Skills, 4 Domain Clusters\nExact Match + Cluster Affinity", "#e6fffa", "#319795")
draw_box(ax, 0.68, 0.39, 0.27, 0.10, "D. N-Gram Jaccard (5%)", "Lexical Token Overlap (IoU)", "#e6fffa", "#319795")

# Layer 4: Calibration & Decision Layer
draw_box(ax, 0.36, 0.18, 0.26, 0.15, "Mathematical Calibration", "Floor Rescaling: (Raw - 0.22) / 0.78\nDynamic Exp: +15% * sqrt(Rel)\nTransfer Matrix: +4% to +20%", "#faf5ff", "#805ad5")
draw_box(ax, 0.68, 0.18, 0.27, 0.15, "Gatekeeper & Decision Logic", "Hard 0.0% Non-Tech Gatekeeper\nThresholds: >=65% Highly Suitable,\n>=45% Suitable, <45% Low Match", "#fff5f5", "#e53e3e")

# Output Layer
draw_box(ax, 0.36, 0.02, 0.59, 0.10, "Recruiter Dashboard & Recommendation (Flask Web UI)", "Ranked Shortlist, Match %, Suitability Badges, Skill Match Inspection, Recruiter Feedback Loop", "#f0fff4", "#38a169")

# Connectors
draw_arrow(ax, 0.30, 0.79, 0.36, 0.81)
draw_arrow(ax, 0.30, 0.59, 0.36, 0.65)
draw_arrow(ax, 0.30, 0.59, 0.36, 0.50)

draw_arrow(ax, 0.62, 0.81, 0.68, 0.82)
draw_arrow(ax, 0.62, 0.81, 0.68, 0.70)
draw_arrow(ax, 0.62, 0.50, 0.68, 0.82)
draw_arrow(ax, 0.62, 0.50, 0.68, 0.57)
draw_arrow(ax, 0.62, 0.81, 0.68, 0.44)

draw_arrow(ax, 0.815, 0.39, 0.50, 0.33)
draw_arrow(ax, 0.62, 0.25, 0.68, 0.25)
draw_arrow(ax, 0.815, 0.18, 0.65, 0.12)

plt.tight_layout()

out_arch = "system_architecture_diagram.png"
plt.savefig(out_arch, dpi=300, bbox_inches='tight')
print(f"System architecture diagram saved to {out_arch}")

artifact_dir = r"C:\Users\Kavinda\.gemini\antigravity\brain\bdd49f83-8001-4302-b026-c1256c609d6d"
if os.path.exists(artifact_dir):
    shutil.copyfile(out_arch, os.path.join(artifact_dir, out_arch))
    print(f"Copied architecture diagram to artifact directory.")
