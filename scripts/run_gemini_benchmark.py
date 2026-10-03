import pandas as pd
import json
import os
import sys
import time
from tqdm import tqdm
import google.generativeai as genai

# ==============================================================================
# IMPORTANT: Put your free Gemini API Key here to run the benchmark!
# Get one at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "PUT_YOUR_API_KEY_HERE"
# ==============================================================================

if GEMINI_API_KEY == "PUT_YOUR_API_KEY_HERE":
    print("❌ Error: You must insert your Gemini API Key at the top of this script first.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
# Using flash model for high speed and low cost
model = genai.GenerativeModel('gemini-1.5-flash')

# 1. Add web_app to path to import your V4.1 local engine
sys.path.insert(0, os.path.abspath('web_app'))
try:
    from nlp_engine import calculate_match_scores
except ImportError:
    print("❌ Could not import nlp_engine. Run this script from the project root.")
    sys.exit(1)

print("Loading 1,000 real resumes from benchmark_1000_sample.csv...")
df = pd.read_csv('benchmark_1000_sample.csv')

TEST_JOB = """
Role: Senior Python & AI Engineer
Required Experience: 3+ years
Must have: Python, PyTorch or TensorFlow, Natural Language Processing, REST APIs (FastAPI/Flask), Docker, Linux.
"""

PROMPT_TEMPLATE = f"""
You are an expert IT Recruiter. Evaluate this resume against the following job description.

JOB DESCRIPTION:
{TEST_JOB}

RESUME:
{{resume_text}}

Does this candidate possess the required technical skills and experience to pass a technical screen for this role?
Answer with EXACTLY ONE WORD: "PASS" or "FAIL". Do not provide any other explanation.
"""

results = []

print(f"\n🚀 Running 1,000 Resumes through V4.1 Engine AND Gemini AI...")
print(f"Note: This will take a few minutes. Gemini API allows 15 requests per minute on the free tier, so we will batch and sleep if necessary.")

for index, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
    text = str(row['Resume'])
    
    # --- 1. Score with your V4.1 Local Engine ---
    local_res = calculate_match_scores(TEST_JOB, [text])[0]
    local_score = local_res['score']
    local_pass = local_score >= 38.0  # Our "Suitable" threshold
    
    # --- 2. Score with Gemini AI Ground Truth ---
    gemini_pass = False
    prompt = PROMPT_TEMPLATE.format(resume_text=text[:3000]) # truncated to save tokens/speed
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            verdict = response.text.strip().upper()
            gemini_pass = "PASS" in verdict
            break
        except Exception as e:
            if "429" in str(e): # Rate limit hit
                time.sleep(10)
            else:
                print(f"Gemini API Error: {e}")
                break
                
    results.append({
        "Category": row['Category'],
        "V4_1_Score": local_score,
        "V4_1_Pass": local_pass,
        "Gemini_Pass": gemini_pass
    })
    
    # Sleep to respect Gemini Free Tier 15 RPM limits (1 req / 4 secs)
    time.sleep(4)

# --- Calculate Metrics ---
res_df = pd.DataFrame(results)

true_positives = len(res_df[(res_df['V4_1_Pass'] == True) & (res_df['Gemini_Pass'] == True)])
false_positives = len(res_df[(res_df['V4_1_Pass'] == True) & (res_df['Gemini_Pass'] == False)])
true_negatives = len(res_df[(res_df['V4_1_Pass'] == False) & (res_df['Gemini_Pass'] == False)])
false_negatives = len(res_df[(res_df['V4_1_Pass'] == False) & (res_df['Gemini_Pass'] == True)])

total_gemini_passes = true_positives + false_negatives
total_gemini_fails = true_negatives + false_positives

print("\n" + "="*50)
print("🏆 V4.1 ENGINE vs. GEMINI AI (GROUND TRUTH) RESULTS")
print("="*50)
print(f"Total Resumes Audited: {len(res_df)}")
print(f"Gemini deemed {total_gemini_passes} as PASS and {total_gemini_fails} as FAIL.\n")

print(f"✅ True Positives (Both agreed to Pass): {true_positives}")
print(f"✅ True Negatives (Both agreed to Reject): {true_negatives}")
print(f"❌ False Positives (V4.1 Passed, but Gemini Rejected): {false_positives}")
print(f"❌ False Negatives (V4.1 Rejected, but Gemini Passed): {false_negatives}\n")

if total_gemini_passes > 0:
    recall = (true_positives / total_gemini_passes) * 100
    print(f"🎯 Target Hit Rate (Recall): {recall:.2f}%")
else:
    print("🎯 Target Hit Rate (Recall): N/A (Gemini rejected all)")

if total_gemini_fails > 0:
    fpr = (false_positives / total_gemini_fails) * 100
    print(f"🛡️ False Positive Rate: {fpr:.2f}%")
    
print("="*50)
res_df.to_csv("gemini_vs_v4_1_results.csv", index=False)
print("Detailed results saved to gemini_vs_v4_1_results.csv")
