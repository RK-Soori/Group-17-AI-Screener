# MASTER INSTRUCTIONS FOR CLAUDE: UPDATE `Progress Review 01.docx`

---

## 🎯 YOUR OBJECTIVE
You are an expert academic technical writer and AI researcher. Your task is to update and upgrade the attached Microsoft Word report:
📄 **`Progress Review 01.docx`**

You must update this document into a finalized, high-scoring, publication-grade academic report for:
* **Course:** Essentials of Artificial Intelligence (Group Assignment)
* **Stage:** Stage 2: Progress Review (Due Week 8 | Weight: 10%)
* **Institution:** General Sir John Kotelawela Defence University (KDU), Faculty of Computing
* **Group:** Group Number 17

---

## 📂 ATTACHED ASSETS IN THIS FOLDER
When executing this prompt, use the following files provided in this directory:
1. **`Progress Review 01.docx`** — The original draft Word document. You MUST keep its exact fonts, cover page, table styles, heading hierarchy, margins, and section numbering.
2. **`STAGE_2_PROGRESS_REVIEW_SUBMISSION.md`** — The comprehensive reference document containing all updated text, tables, formulas, and data.
3. **`system_architecture_diagram.png`** — High-resolution (300 DPI) system architecture diagram (Figure 1).
4. **`benchmark_comparison_graph.png`** — 3-panel high-resolution comparison chart against Gemini ground truth (Figure 2).
5. **`v3_real_benchmark_comparison.png`** — 3-panel high-resolution comparison chart on the 13,389 real resume dataset (Figure 3).
6. **`STAGE2_PROGRESS_SHOWCASE.html`** — Standalone recruiter UI dashboard with live candidate scorecards and suitability badges.

---

## ⚠️ CRITICAL RULES FOR UPDATING THE DOCUMENT
1. **PRESERVE FORMATTING & STRUCTURE:** Maintain the exact template, headers, footers, cover page, table of contents structure, font sizes, and line spacing of `Progress Review 01.docx`.
2. **DELETE DRAFT PLACEHOLDERS:**
   * Remove any placeholder notes, especially the Sinhala text on line 209:
     `web application එකේ screenshots තියෙනවා නම් CV upload page + job input page + result/ranking page screenshots`.
   * Replace this placeholder with the actual benchmark evaluation tables and figure references provided below.
3. **UPDATE OUTDATED BASELINE METRICS:**
   * In the draft, early baseline metrics were reported (66.67% accuracy, 14.04% MAE, 0.9700 correlation).
   * You MUST update these to the finalized **Version 3.0 metrics**:
     * **Classification Label Accuracy:** **100.00% (12/12 exact matches)**
     * **Top-1 Candidate Ranking Accuracy (Precision@1):** **100.00% (3/3 correct #1 picks)**
     * **Mean Absolute Error (MAE):** **4.81%** (All-time best, dropped from 14.04%)
     * **Pearson Correlation ($r$):** **0.9900** (Near-perfect alignment with Gemini ground truth)
     * **Non-Technical False Positive Rate:** **0.00%** (0 false positives out of 60 real non-tech resumes)
     * **Statistical Discrimination Gap:** **+39.03%** separation between qualified and non-tech profiles.

---

## 📝 SECTION-BY-SECTION EXACT EDITING INSTRUCTIONS

### Cover Page & Metadata
* **Keep Intact:** Title, Group 17, and the 4 student details:
  * R.T.I.K. Prabodhani (`D/DBA/25/0010` - BSc Hons in Data Science & Business Analytics)
  * K.V.P.M. Sewmini (`D/BIS/24/0012` - BSc Hons in Information Systems)
  * O.K.D.A. Dilmira (`D/BIT/24/0055` - BSc Hons in Information Technology)
  * Kavinda Sooriyarachchi (`D/COE/25/0016` - BSc Hons in Computer Engineering)

---

### Section 1: Updated Project Title and Problem Statement
* **1.1 Updated Project Title:**
  Update to:  
  **"AI-Based Job Candidate Screening and Recommendation System Using a Hybrid Fine-Tuned Semantic Transformer and Domain-Adapted N-Gram Architecture"**
* **1.2 Updated Problem Statement:**
  Expand the problem statement by explicitly articulating the three core technical challenges solved:
  1. *Universal English Cosine Overlap:* Off-the-shelf NLP models assign ~22% match to non-technical applicants (e.g., Chefs scoring 25% for AI roles) due to common English syntax.
  2. *Middle-Tier Adjacent Penalty:* Rigid keyword systems penalize transferable technical profiles (e.g., Data Analyst $\to$ AI, Backend $\to$ Frontend).
  3. *The 256-Token Truncation Barrier:* Pre-trained transformers truncate resumes at ~200 words, discarding Page 2 project experience in real-world multi-page resumes.

---

### Section 2: Changes Made After Proposal Feedback
Keep subsections 2.1 to 2.4, and add subsections 2.5 through 2.8:
* **2.5 Baseline Floor Calibration Layer:** Subtraction of the universal 0.22 background cosine floor to eliminate non-technical false positives.
* **2.6 Cross-Domain Career Transferability Matrix:** Bounded adjacency boosts (+20% Data Analyst $\to$ AI, +7.5% Backend $\to$ Frontend, +4% Sysadmin $\to$ DevOps).
* **2.7 Sliding-Window Document Chunking with Max-Pooling:** Overcomes 256-token truncation on multi-page resumes by chunking text into 140-word overlapping windows.
* **2.8 Multi-Span Date Range Tenure Parser:** Extracts total cumulative career tenure from employment dates (`2018 - 2022`, `2020 - Present`), replacing fragile regex.

---

### Section 3: Finalized AI Techniques
* **Update Section 3.6 (Hybrid Scoring Engine):**
  Replace the preliminary scoring formula with the finalized **Version 3.0 Ensemble Formula**:
  $$\text{Raw Blended} = 0.45 \cdot \text{Sim}_{\text{SBERT}} + 0.35 \cdot \text{Sim}_{\text{TF-IDF}} + 0.15 \cdot \text{Sim}_{\text{Skill}} + 0.05 \cdot \text{Sim}_{\text{Jaccard}}$$
  $$\text{Score}_{\text{calibrated}} = \max\left(0.0, \frac{\text{Raw Blended} - 0.22}{1.0 - 0.22}\right) \times 100$$
  $$\text{Experience Bonus} = 15.0 \times \min\left(1.0, \sqrt{\frac{\text{Score}_{\text{calibrated}}}{100}}\right)\quad (\text{if Tenure}_{\text{cand}} \ge \text{Tenure}_{\text{req}})$$
  $$\text{Technical Gatekeeper:} \quad \text{If } |\text{Candidate Skills}| == 0 \implies \text{Final Score} = 0.0\%$$
  Explain the decision thresholds:
  * **Highly Suitable:** $\ge 65.0\%$ (Direct shortlist)
  * **Suitable:** $45.0\% - 64.9\%$ (Transferable / interview candidate)
  * **Low Match:** $< 45.0\%$ (Automated rejection / archive)

---

### Section 4: System Architecture / Workflow
* **Insert Image:** Embed **`system_architecture_diagram.png`** as **Figure 1: End-to-End System Architecture of Hybrid Screening Engine (v3.0)**.
* **Explain the 5 Layers:**
  1. *Layer 1 (Input):* Job description and multi-format candidate resumes.
  2. *Layer 2 (Preprocessing):* HR stopword filtering, multi-span date parsing, and 140-word sliding-window chunking.
  3. *Layer 3 (Matching Engines):* Fine-tuned SBERT (45%), Domain TF-IDF (35%), Bounded Skill Taxonomy (15%), N-Gram Jaccard (5%).
  4. *Layer 4 (Calibration & Rules):* 0.22 floor rescaling, dynamic square-root experience bonus, career transferability matrix, and technical gatekeeper.
  5. *Layer 5 (Decision & UI):* Decision tier classification and recruiter dashboard.

---

### Section 5: Dataset Details
Retain the existing descriptions of `UpdatedResumeDataSet.csv` (962 resumes) and job descriptions (1,068 jobs), and ADD:
1. **Domain TF-IDF Pre-Fitting Corpus:** 2,076 documents learning **48,393 unique n-grams** (unigrams and bigrams).
2. **Real-World Scalability Corpus (`Resume_Dataset_Real.csv`):** **13,389 real resumes** across **43 diverse categories** (62.21 MB).
3. **Ground Truth Benchmark Suite:** 12 candidate profiles evaluated against Google Gemini LLM ground truth across 3 technical domains.

---

### Section 6: Rules and AI Model Structure
* **6.1 Rule-Based Component:**
  Detail the dynamic experience bonus scaled by $\sqrt{\text{Relevance}}$, the career transferability pairs, and the technical gatekeeper rule.
* **6.2 Neural Network / SBERT Structure:**
  Include the sliding-window chunking max-pooling formula:
  $$\text{Sim}_{\text{SBERT}} = 0.70 \times \max_{k} \left(\cos(\vec{J}, \vec{C}_k)\right) + 0.30 \times \text{Mean}\left(\text{Top-2}\left(\cos(\vec{J}, \vec{C}_k)\right)\right)$$
  and state the training parameters: 6 Transformer encoder layers, 384 hidden dimensions, Multiple Negatives Ranking Loss (MNRL) with temperature $\tau = 0.05$, trained on an NVIDIA RTX 4050 GPU in 55.6 seconds ($r = 0.9085$).

---

### Section 7: Partial Implementation / Current Results
* **DELETE the Sinhala placeholder text:** Remove line 209 completely.
* **Insert the 12-Candidate Benchmark Results Table:**

| Job Opening | Candidate Name | Candidate Background | System Score | System Label | Gemini Target | Gemini Label | Match Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Python & AI** | Alex Chen | Senior AI Engineer | **89.3%** | Highly Suitable | 90.0% | Highly Suitable | **CORRECT (OK)** |
| **Python & AI** | Sarah Miller | Data Analyst (Transferable) | **48.2%** | Suitable | 55.0% | Suitable | **CORRECT (OK)** |
| **Python & AI** | David Clark | Junior Web Developer | **25.6%** | Low Match | 30.0% | Low Match | **CORRECT (OK)** |
| **Python & AI** | Marcus Vance | Executive Head Chef | **0.0%** | Low Match | 5.0% | Low Match | **CORRECT (OK)** |
| **Frontend React**| Elena Rostova | React Developer | **90.7%** | Highly Suitable | 88.0% | Highly Suitable | **CORRECT (OK)** |
| **Frontend React**| Kevin Patel | Backend Java (Transferable)| **45.2%** | Suitable | 48.0% | Suitable | **CORRECT (OK)** |
| **Frontend React**| Lisa Wong | UI/UX Designer | **40.6%** | Low Match | 32.0% | Low Match | **CORRECT (OK)** |
| **Frontend React**| Robert Taylor | Chartered Accountant | **0.0%** | Low Match | 5.0% | Low Match | **CORRECT (OK)** |
| **DevOps & Cloud** | Tariq Mansoor | Senior DevOps Engineer | **88.4%** | Highly Suitable | 92.0% | Highly Suitable | **CORRECT (OK)** |
| **DevOps & Cloud** | Brian Adams | Linux Sysadmin (Transferable)| **51.1%** | Suitable | 52.0% | Suitable | **CORRECT (OK)** |
| **DevOps & Cloud** | Emily Watson | Technical Support | **15.8%** | Low Match | 28.0% | Low Match | **CORRECT (OK)** |
| **DevOps & Cloud** | James Sullivan| Civil Project Manager | **0.0%** | Low Match | 5.0% | Low Match | **CORRECT (OK)** |

* **Insert Image:** Embed **`benchmark_comparison_graph.png`** as **Figure 2: Benchmark Evaluation Results: System v3.0 vs. Gemini Ground Truth (100% Accuracy, MAE 4.81%)**.
* **Insert Large-Scale Real Dataset Evaluation Table (120 Resumes across 15 Disciplines):**

| Discipline Tier | Candidate Profession | v2.2 Mean | v3.0 Mean | Empirical Delta | Non-Tech FPR | Zero-Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Target Technical** | Web Designing | 28.36% | **37.33%** | +8.97% | 0.0% | 0.0% |
| **Target Technical** | DevOps | 47.99% | **54.08%** | +6.10% | 0.0% | 0.0% |
| **Adjacent Technical**| Database Administrator | 5.99% | **11.73%** | +5.75% | 0.0% | 0.0% |
| **Target Technical** | Data Science | 26.83% | **32.48%** | +5.65% | 0.0% | 0.0% |
| **Adjacent Technical**| Java Developer | 27.38% | **30.50%** | +3.12% | 0.0% | 0.0% |
| **Adjacent Technical**| DotNet Developer | 35.14% | **38.21%** | +3.07% | 0.0% | 0.0% |
| **Target Technical** | Python Developer | 29.89% | **32.46%** | +2.57% | 0.0% | 0.0% |
| **Adjacent Technical**| Network Security | 14.15% | **16.78%** | +2.63% | 0.0% | 0.0% |
| **Target Technical** | React Developer | 56.99% | **57.34%** | +0.36% | 0.0% | 0.0% |
| **Non-Technical** | Accountant | 2.00% | **2.00%** | +0.00% | **0.00%** | 90.0% at 0.0% |
| **Non-Technical** | Civil Engineer | 1.67% | **1.46%** | -0.21% | **0.00%** | 90.0% at 0.0% |
| **Non-Technical** | Sales Professional | 0.60% | **0.60%** | +0.00% | **0.00%** | 95.0% at 0.0% |
| **Non-Technical** | Food & Beverages (Chef) | 0.00% | **0.00%** | +0.00% | **0.00%** | **100% Zeroed** |
| **Non-Technical** | Human Resources | 0.00% | **0.00%** | +0.00% | **0.00%** | **100% Zeroed** |
| **Non-Technical** | Arts Professional | 0.00% | **0.00%** | +0.00% | **0.00%** | **100% Zeroed** |

* **Insert Image:** Embed **`v3_real_benchmark_comparison.png`** as **Figure 3: Real-World Dataset Performance Comparison (Discrimination Gap: +39.03%, Non-Tech FPR: 0.00%)**.

---

### Section 8: Problems Faced During Development
* Keep subsections 8.1 (OCR), 8.2 (TF-IDF corpus), and 8.3 (SBERT dependencies).
* **Update 8.4 ("Over-Scoring Problem"):**
  State clearly that this problem was **successfully resolved** by implementing baseline floor subtraction ($0.22$) and the dynamic square-root experience bonus.
* **Add 8.5 ("The 256-Token Sequence Truncation Barrier"):**
  Describe how standard SBERT truncated 800-word resumes, discarding Page 2 projects, and how sliding-window chunking with max-pooling solved it.
* **Add 8.6 ("Date-Range Tenure Extraction Failure"):**
  Describe how candidates write employment dates (*"2019 - 2023"*) rather than explicit years, and how the multi-span date parser solved it.

---

### Section 9: Remaining Work Before Final Submission
* Update the remaining work to reflect what has **already been completed** in Version 3.0 vs. what remains for final submission:
  * *Completed in v3.0:* Dynamic experience bonus, threshold recalibration (65%/45%), sliding-window chunking, 143+ skill taxonomy, transferability matrix, real dataset validation.
  * *Remaining for Final Submission (Weeks 9–14):*
    1. **Database Persistence Integration (MySQL):** Storing job requisitions, candidate profiles, and historical scoring data.
    2. **Native Multi-Format Parser:** Direct PDF and DOCX drag-and-drop parsing using `pdfplumber` and `python-docx`.
    3. **Recruiter Dashboard Enhancements:** Side-by-side candidate comparison views and CSV export.
    4. **Active Learning Feedback Loop:** Storing recruiter decisions to refine model weights.
    5. **System Load Testing & Viva Presentation Preparation.**

---

### Section 10: Updated Contribution of Each Member
Ensure all 4 members have clear, balanced technical contributions:
* **Kavinda Sooriyarachchi (`D/COE/25/0016`):** SBERT GPU fine-tuning (RTX 4050 GPU, MNRL loss); sliding-window chunking implementation; mathematical baseline floor calibration; backend Flask web engine integration.
* **R.T.I.K. Prabodhani (`D/DBA/25/0010`):** Rule-based recruitment requirements; Technical Gatekeeper filter logic; Career Transferability Matrix rules; benchmark error analysis.
* **K.V.P.M. Sewmini (`D/BIS/24/0012`):** Multi-Span Date Range Tenure Parser; acquisition and partitioning of the 13,389 Kaggle real resume dataset; category-by-category empirical evaluation.
* **O.K.D.A. Dilmira (`D/BIT/24/0055`):** Domain TF-IDF model pre-fitting on 2,076 documents (48,393 n-grams); Flask web UI and interactive HTML showcase dashboard; 300 DPI benchmark visualization charts.
* **All Members:** Joint testing, Gemini benchmark validation, Stage 2 progress report preparation, and viva demonstration rehearsal.

---

### Section 11: Current Technology Stack
Ensure the stack includes:
* **Programming Language:** Python 3.12
* **Machine Learning / Transformers:** SentenceTransformers, PyTorch (NVIDIA GeForce RTX 4050 GPU acceleration)
* **Lexical NLP:** Scikit-learn (TfidfVectorizer), Regex
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Flask, HTML5, CSS3, JavaScript
* **Development & Version Control:** Visual Studio Code, Git & GitHub

---

### Section 12 & 13: Progress Summary and Conclusion
* Update the progress summary and conclusion to reflect that the system has successfully achieved **100.00% label accuracy**, **100.0% Top-1 precision**, **4.81% MAE**, and validated scalability against a massive **13,389 real-world resume dataset** with a **+39.03% discrimination gap** and **0.00% false positive rate**.

---

## 🚀 EXECUTION INSTRUCTION FOR CLAUDE
Now, open and process `Progress Review 01.docx`, apply all the modifications outlined above, embed the image figures at their respective locations, update the Table of Contents, and output the updated document as:
📄 **`Progress_Review_01_Updated_Final.docx`**
