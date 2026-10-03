import os
import sys
import pandas as pd
import json

# Add web_app to path to import nlp_engine
sys.path.insert(0, os.path.abspath('web_app'))
from nlp_engine import calculate_match_scores

print("=" * 80)
print("EVALUATING V4.1 AI ENGINE ON 4TH DATASET (ganchengguang/resume-5label-classification)")
print("=" * 80)

csv_path = 'fourth_resume_dataset.csv'
df = pd.read_csv(csv_path)

# Sample 500 random resumes from the 40,000
df_sample = df.sample(n=500, random_state=42).reset_index(drop=True)

# Define our benchmark test
TEST_JOB = {
    "role": "Senior Python & AI Engineer",
    "description": "Looking for a Senior Python & AI Engineer with at least 3 years of experience. Must be proficient in Python, PyTorch or TensorFlow, Natural Language Processing, building REST APIs with FastAPI or Flask, and containerization using Docker and Linux."
}

results = []

print(f"Testing Job: {TEST_JOB['role']}")
print("Running blind inference on 500 random resumes...")

# Extract text column
resume_texts = df_sample['text'].astype(str).tolist()

# Run batch inference (nlp_engine takes the whole list)
try:
    scores = calculate_match_scores(TEST_JOB["description"], resume_texts)
    
    for res in scores:
        results.append({
            "score": float(res['score']),
            "label": res['label']
        })
except Exception as e:
    print(f"Error during batch inference: {e}")
    # Fallback to single inference if memory issue
    for text in resume_texts:
        res = calculate_match_scores(TEST_JOB["description"], [text])
        results.append({
            "score": float(res[0]['score']),
            "label": res[0]['label']
        })

eval_df = pd.DataFrame(results)

summary_by_label = eval_df['label'].value_counts().reset_index()
summary_by_label.columns = ['Classification Label', 'Count']
summary_by_label['Percentage'] = (summary_by_label['Count'] / 500) * 100

print("\n" + "=" * 80)
print("FOURTH DATASET (BLIND FIELD TEST) RESULTS")
print("=" * 80)
print(summary_by_label.to_string(index=False))

print("\nScore Statistics:")
print(f"Mean Score: {eval_df['score'].mean():.2f}")
print(f"Median Score: {eval_df['score'].median():.2f}")
print(f"Max Score: {eval_df['score'].max():.2f}")
print(f"Min Score: {eval_df['score'].min():.2f}")

zero_score_rate = (eval_df['score'] == 0.0).mean() * 100
print(f"\nHard Zero Rate (Completely Irrelevant Resumes): {zero_score_rate:.2f}%")
print("-" * 50)
