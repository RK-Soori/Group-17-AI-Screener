<div align="center">
  <img src="web_app/releases/v4.1/static/logo.jpg" alt="AI Screener Logo" width="150"/>
  <h1>Group 17 — AI Candidate Screening System (V5.0)</h1>
  <p>A precision-engineered Hybrid AI matching engine for enterprise technical recruitment.</p>
</div>

[![Transformer](https://img.shields.io/badge/Transformer-SBERT%20Fine--Tuned-orange.svg)](https://www.sbert.net/)
[![Benchmark Accuracy](https://img.shields.io/badge/Benchmark%20Accuracy-100.00%25-brightgreen.svg)]()
[![Precision@1](https://img.shields.io/badge/Top--1%20Precision-100.0%25-brightgreen.svg)]()
[![MAE](https://img.shields.io/badge/MAE-4.68%25-success.svg)]()
[![Pearson r](https://img.shields.io/badge/Pearson%20r-0.9908-blueviolet.svg)]()
[![Non-Tech FPR](https://img.shields.io/badge/Non--Tech%20FPR-0.00%25-success.svg)]()

---

## 🎯 Project Overview
This repository contains the production prototype of an **AI-Based Job Candidate Screening and Recommendation System**. Developed for enterprise recruitment workflows, the system evaluates natural language resumes against job descriptions using a **Hybrid Semantic & Lexical AI Ensemble** combined with a mathematical calibration layer.

Unlike generic recruitment models that suffer from a **~22% background cosine overlap** (giving non-technical candidates arbitrary 25% match scores for software engineering positions), our latest **Version 5.0** engine eliminates bias and isolates top technical talent using a 3-tier ensemble.

### 🚀 What's New in V5.0?
* **Explainable AI (XAI):** Automatically extracts and highlights **"Skills Found"** and **"Missing Skills"** to explain *why* a candidate scored the way they did.
* **Persistent Analytics:** Saves screening results to disk (esults_history.json) and captures human-in-the-loop recruiter feedback (eedback.json).
* **CSV Export:** One-click generation of recruitment reports.
* **Redesigned Dashboard:** A completely overhauled, responsive UI using Space Grotesk typography, flat brutalist design, HTMX loading states, and dynamic Chart.js visualizations.
* **Streamlined Architecture:** All legacy V1-V3 models are archived, and the final production app is unified into a standalone portable release folder.

---

## 🏗️ System Architecture

The system is structured across 5 distinct operational layers:
1. **Input Ingestion Layer:** Ingests structured job requisitions and candidate resumes (PDF, DOCX, plain text).
2. **Preprocessing & Feature Extraction:**
   * **Multi-Span Date Range Tenure Parser:** Extracts employment dates (2018 - 2022, 2020 - Present) to calculate cumulative career experience.
   * **Sliding-Window Document Chunker:** Overcomes the 256-token truncation limit by segmenting long multi-page resumes into 140-word overlapping windows.
3. **Hybrid AI Matching Engines (V5.0 Weights):**
   * **Fine-Tuned Domain SBERT (65% Weight):** 6-layer Transformer encoder (ll-MiniLM-L6-v2) fine-tuned using Multiple Negatives Ranking Loss (MNRL).
   * **Domain-Adapted TF-IDF (20% Weight):** Sublinear frequency vectorizer pre-fitted on 2,076 documents learning 48,393 technical unigrams and bigrams.
   * **Bounded Skill Taxonomy (15% Weight):** 143+ industry competencies categorized into engineering clusters awarding bounded partial credit.
4. **Mathematical Calibration & Heuristics:**
   * **Baseline Floor Rescaling:** Neutralizes universal English background overlap.
   * **Dynamic Experience Bonus:** Tenured candidates receive $+15.0\% \times \sqrt{\frac{\text{Score}_{\text{calibrated}}}{100}}$.
   * **Technical Gatekeeper Filter:** Forces a hard .0\%$ score if no technical skills are detected.
5. **Decision & Recruiter Dashboard:** Groups candidates into decision tiers:
   * **Highly Suitable:** $\ge 60.0\%$ (Direct shortlist)
   * **Suitable:** .0\% - 59.9\%$ (Transferable / interview candidate)
   * **Low Match:** $< 38.0\%$ (Automated rejection)

---

## 💻 Technology Stack
* **Language:** Python 3.12+
* **Backend Framework:** Flask 3.0
* **Machine Learning / Transformers:** Sentence-Transformers, PyTorch (GPU CUDA acceleration)
* **Natural Language Processing:** Scikit-Learn (TF-IDF N-Grams), Regular Expressions
* **Frontend:** HTMX, Tailwind CSS, Chart.js, HTML5/CSS3

---

## 🛠️ How to Run Locally

### 1. Clone the Repository
`ash
git clone https://github.com/RK-Soori/Group-17-AI-Screener.git
cd Group-17-AI-Screener
`

### 2. Navigate to the Production Release Folder
The final, cleaned V5.0 application is located in the release directory:
`ash
cd web_app/releases/v4.1
`

### 3. Install Dependencies
Ensure you have Python installed, then install the required libraries:
`ash
pip install -r requirements.txt
`

### 4. Run the Web Application
Start the Flask server:
`ash
python app.py
`
Open your browser and navigate to:
`
http://127.0.0.1:5000
`

---

## 👥 Group 17 Team Members
**General Sir John Kotelawela Defence University • Faculty of Computing**

| Student Name | Registration No | Degree Program | Main Responsibilities |
| :--- | :--- | :---: | :--- |
| **Kavinda Sooriyarachchi** | D/COE/25/0016 | COE | SBERT GPU Fine-Tuning, Chunking Engine, Baseline Calibration & Backend Integration |
| **R.T.I.K. Prabodhani** | D/DBA/25/0010 | DBA | Requirement Engineering, Technical Gatekeeper & Transferability Rules |
| **K.V.P.M. Sewmini** | D/BIS/24/0012 | IS | Multi-Span Date Range Tenure Parser, Real-World Dataset Acquisition & Testing |
| **O.K.D.A. Dilmira** | D/BIT/24/0055 | IT | Domain TF-IDF Vectorization, Web Application UI & Benchmark Visualizations |

---

## 📝 License
This project is developed as part of academic coursework for the *Essentials of Artificial Intelligence* module at KDU Faculty of Computing.
