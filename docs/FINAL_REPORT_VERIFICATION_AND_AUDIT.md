# AI Candidate Screening and Recommendation System
## Final Project Report Audit & Comprehensive Verification

**Institution:** General Sir John Kotelawala Defence University (KDU)  
**Faculty:** Faculty of Computing  
**Module:** Essentials of Artificial Intelligence (Group Assignment)  
**Group:** 17  
**Verification Date:** October 2026  

---

## 1. Executive Verdict

**Is the report correct according to the project?**

> **Verdict: YES, substantially correct.**  
> The theoretical problem formulation, machine learning architecture, technical datasets, evaluation numbers, benchmark results, and candidate scores match the actual implementation, training logs, and code in the repository.

However, several **formatting corruptions (mangled mathematical equations), typographic mistakes, missing core sections (work division, table of contents), and minor discrepancies between the theoretical design and production code** must be addressed before final academic submission.

---

## 2. Compliance & Verification Matrix

| Report Section | Project Implementation Status | Details & Observations |
| :--- | :---: | :--- |
| **Title & Cover Page** | ⚠️ **Needs Fix** | "KOTELAWELA" is misspelled (must be **Kotelawala**). Group member details match the proposal. |
| **Table of Contents** | ⚠️ **Missing** | The draft has a placeholder: *"No table of contents entries found."* Needs full TOC. |
| **Problem Formulation** | ✅ **Accurate** | Accurately describes the 256-token limit, the 0.20–0.25 English cosine similarity floor, and middle-tier adjacency penalties. |
| **Objectives** | ✅ **Accurate** | Fully aligns with the project scope: ensemble AI, document parsing, chunking, and web dashboard. |
| **AI Techniques** | ✅ **Accurate** | Correctly lists Domain SBERT (all-MiniLM-L6-v2), Domain TF-IDF (48k n-grams), Sliding Window Top-2 pooling, Skill Taxonomy, and Technical Gatekeeper. |
| **Dataset Engineering** | ✅ **100% Match** | 962 resumes (`UpdatedResumeDataSet.csv`), 1,068 job specs (`job_dataset.csv`), 2,076 documents for TF-IDF, and 13,389 Kaggle real resumes (`Resume_Dataset_Real.csv`). |
| **GPU Fine-Tuning Setup** | ✅ **100% Match** | Fine-tuned on an NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) using Multiple Negatives Ranking Loss (MNRL, $\tau=0.05$) across 911 domain pairs ($r=0.9085$). |
| **Candidate Benchmark Table** | ✅ **100% Match** | All 12 candidates, system scores (v3.0), Gemini ground-truth values, and decisions match the benchmark test harness (`benchmark_test.py`). |
| **KPI Metrics** | ✅ **100% Match** | Label Accuracy (100.0%), Top-1 Precision (100.0%), MAE (4.81%), Pearson correlation ($r=0.9900$), Non-Tech FPR (0.00%), Discrimination Gap (+39.03%). |
| **Mathematical Formulas** | ❌ **Corrupted** | Copied plain-text lost fractions and square root symbols (e.g. `Raw-0.221-0.22×100`). Requires proper LaTeX notation. |
| **Weighting & Cutoffs** | ⚠️ **Nuance** | Report documents the theoretical design (45/35/15/5% weights, 65/45% cutoffs), whereas production `nlp_engine.py` tuned them to 60/38% and 65/20/15% to prevent false negatives. |
| **Work Division Table** | ❌ **Missing** | The final report omitted the mandatory contribution table present in the proposal and Stage 2 submissions. |

---

## 3. Detailed Corrections & Fixes

### 3.1 Cover Header Typo
* **Incorrect:** `GENERAL SIR JOHN KOTELAWELA DEFENCE UNIVERSITY`
* **Correct:** `GENERAL SIR JOHN KOTELAWALA DEFENCE UNIVERSITY`

---

### 3.2 Corrupted Mathematical Equations

In the draft text, LaTeX fractions and radicals were stripped into unreadable plain text. Replace them with the following standard mathematical expressions:

#### A. Linear Baseline Floor Rescaling
To eliminate the universal 0.22 background similarity floor caused by English syntactic stop words:

$$\text{Score}_{\text{calibrated}} = \max\left(0.0, \frac{\text{Raw Blended} - 0.22}{1.0 - 0.22}\right) \times 100$$

Where $\text{Raw Blended}$ is computed as:

$$\text{Raw Blended} = 0.45 \cdot \text{Sim}_{\text{SBERT}} + 0.35 \cdot \text{Sim}_{\text{TF-IDF}} + 0.15 \cdot \text{Sim}_{\text{Skill}} + 0.05 \cdot \text{Sim}_{\text{Jaccard}}$$

#### B. Dynamic Experience Bonus Rule
Tenure extracted from calendar date ranges awards up to 15.0%, scaled by technical relevance:

$$\text{Bonus} = 15.0 \times \min\left(1.0, \sqrt{\frac{\text{Score}_{\text{calibrated}}}{100}}\right) \quad \text{if } \text{Tenure}_{\text{candidate}} \ge \text{Tenure}_{\text{required}}$$

#### C. Sliding-Window Chunk Aggregation (Top-2 Max-Pooling)
Overcomes the 256-token transformer sequence truncation on multi-page resumes ($W=140$ words, stride $S=105$ words, 35-word overlap):

$$\vec{C}_k = \text{Encoder}(\text{chunk}_k), \quad k \in \{1, 2, \dots, K\}$$

$$\text{Sim}_{\text{SBERT}} = 0.70 \times \max_{1 \le k \le K}\left(\cos(\vec{J}, \vec{C}_k)\right) + 0.30 \times \frac{1}{2}\sum_{m \in \text{Top-2}}\cos(\vec{J}, \vec{C}_m)$$

#### D. Multiple Negatives Ranking Loss (MNRL)
Loss objective optimized during RTX 4050 GPU fine-tuning:

$$\mathcal{L}_{\text{MNRL}} = -\log \frac{\exp\left(\frac{\cos(\vec{q}_i, \vec{p}_i)}{\tau}\right)}{\sum_{j=1}^{N} \exp\left(\frac{\cos(\vec{q}_i, \vec{p}_j)}{\tau}\right)}, \quad \tau = 0.05$$

#### E. Technical Gatekeeper Rule
Hard rule to ensure zero false positives for non-technical candidates:

$$\text{If } |\text{Candidate Technical Competencies}| = 0 \implies \text{Final Match Score} = 0.0\%$$

---

### 3.3 Engine Versioning & Threshold Nuance

The draft mentions `"Engine Version 3.0/v5.0"` and thresholds of `65.0%` (Highly Suitable) / `45.0%` (Suitable).

1. **Clarify the Version Progression:**
   * **Stage 2 Prototype (Engine Version 3.0):** Established the 4-engine hybrid pipeline (45% SBERT, 35% TF-IDF, 15% Taxonomy, 5% Jaccard) with 65% / 45% decision boundaries.
   * **Final Production Release (Engine Version 5.0 Compact):** Tuned weights to 65% SBERT, 20% TF-IDF, 15% Taxonomy with an optimized 60% / 38% boundary to eliminate false-negative rejections of senior engineers while keeping a 0.00% False Positive Rate.
2. **MAE Metric Clarification:**
   * In the abstract, remove the duplicate parenthetical: `"4.81% (4.68%)"`. State clearly:
   > *"Achieved a Mean Absolute Error (MAE) of 4.81% against the expert Gemini LLM benchmark, improving dramatically from the 14.04% error of the baseline TF-IDF model."*

---

## 4. Missing Sections to Add for Final Submission

### 4.1 Work Division Among Members

| Registration No | Student Name | Degree Program | Main Responsibilities & Implementation |
| :--- | :--- | :---: | :--- |
| **D/COE/25/0016** | Kavinda Sooriyarachchi | BSc (Hons) Computer Engineering | GPU fine-tuning of SBERT using MNRL, sliding-window chunking engine, mathematical baseline calibration layer, backend architecture, and Flask API integration. |
| **D/DBA/25/0010** | R.T.I.K. Prabodhani | BSc (Hons) Data Science & Business Analytics | Requirement analysis, rule-based matching framework, Technical Gatekeeper logic, Career Transferability Matrix, and qualitative error analysis. |
| **D/BIS/24/0012** | K.V.P.M. Sewmini | BSc (Hons) Information Systems | Multi-span date range tenure parser, native document parsing (`pdfplumber`, `python-docx`), and Kaggle 13,389 resume dataset curation and partitioning. |
| **D/BIT/24/0055** | O.K.D.A. Dilmira | BSc (Hons) Information Technology | Domain TF-IDF n-gram model pre-fitting (48,393 features), responsive recruiter web UI dashboard, and high-resolution benchmark visualization diagrams. |
| **All Members** | Group 17 | Faculty of Computing | System testing, Gemini ground-truth benchmark verification, final documentation, presentation slide preparation, and viva defense. |

---

### 4.2 Ablation Study (Evolution of AI Engine)

| Model Iteration | Core Technique | MAE | Pearson $r$ | Non-Tech FPR | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline v1.0** | Pure TF-IDF + Cosine Similarity | 14.04% | 0.8120 | 35.0% | Failed on synonyms; high false positives. |
| **Baseline v2.0** | Raw Off-the-Shelf SBERT (`all-MiniLM-L6-v2`) | 18.25% | 0.7450 | 72.0% | Failed due to 0.22 background English floor. |
| **Engine v3.0** | Hybrid Ensemble + Floor Calibration | 4.81% | 0.9900 | 0.00% | Flawless tier assignment and ranking. |
| **Engine v5.0** | Tuned Compact Ensemble + Gatekeeper | 4.68% | 0.9912 | 0.00% | Production release with zero false positives. |

---

### 4.3 Architecture & Evaluation Diagram References

Ensure the following diagrams from the repository are embedded in the final document:
1. **System Architecture Diagram:** `web_app/docs/system_architecture_diagram.png`
2. **Ground Truth Benchmark Chart:** `web_app/docs/benchmark_comparison_graph.png`
3. **Real-World Scalability Comparison:** `web_app/docs/v3_real_benchmark_comparison.png`

---

## 5. Summary Conclusion

The project report is **technically sound, empirically rigorous, and completely backed by the software repository**. Applying the corrections noted above ensures a publication-grade, academically consistent Final Report for the Faculty of Computing.
