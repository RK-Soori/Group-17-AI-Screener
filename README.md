# AI-Based Job Candidate Screening System
**Group 17 Project Prototype**

This repository contains the prototype for an AI-Based Job Candidate Screening System. It is designed to automatically evaluate natural language candidate resumes against a natural language Job Description and score their suitability using advanced Natural Language Processing (NLP) techniques.

## 🚀 Features & AI Architecture

Our system goes far beyond standard keyword matching by implementing a **Hybrid AI Matcher** that blends traditional lexical algorithms with modern deep-learning semantic transformers:

1. **Semantic Understanding (Deep Learning):** Utilizes `Sentence-BERT (all-MiniLM-L6-v2)` to understand the context of the resumes. This allows the AI to recognize that "Programmer" and "Software Developer" mean the same thing, even if the exact words don't match.
2. **Lexical Matching (TF-IDF):** Utilizes Term Frequency-Inverse Document Frequency and N-Grams to mathematically guarantee that exact keywords are rewarded.
3. **Keyword Weighting Strategy:** Bypasses TF-IDF limitations by artificially boosting the mathematical weight of critical tech skills (e.g., Python, AWS, SQL) by 5x to mimic human HR screening logic.
4. **Jaccard Similarity:** Blends direct word-overlap percentages into the final hybrid score.
5. **Rule-Based Heuristics:** Detects explicit constraints (e.g., "X years of experience") and applies flat bonuses if the candidate passes the logic check.
6. **Domain Stopwords & Section Heuristics:** Custom Python logic strips out HR fluff words ("passionate", "team player") and truncates irrelevant sections (Hobbies, References) before the math runs.
7. **Human-in-the-loop (HITL):** A web interface with 👍 / 👎 feedback buttons allowing HR to provide input for future reinforcement learning.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **AI / ML Libraries:** scikit-learn, sentence-transformers, PyTorch
* **Frontend:** HTML5, CSS3, Vanilla JS

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "group project ai/web_app"
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.12+ installed.
   ```bash
   pip install flask scikit-learn sentence-transformers
   ```

3. **Start the Flask Server:**
   ```bash
   python app.py
   ```
   *Note: On the very first run, the system will download the ~90MB `all-MiniLM` HuggingFace model to your local cache.*

4. **Test the UI:**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 👥 Authors
* Group 17
