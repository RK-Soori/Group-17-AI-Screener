import pandas as pd
import json
import os, sys

sys.path.insert(0, os.path.abspath('web_app'))
from nlp_engine import calculate_match_scores

TEST_JOB = 'Looking for a Senior Python & AI Engineer with at least 3 years of experience. Must be proficient in Python, PyTorch or TensorFlow, Natural Language Processing, building REST APIs with FastAPI or Flask, and containerization using Docker and Linux.'

print('Loading the 20 STRICTLY REAL Positive resumes...')
df = pd.read_csv('strictly_real_positives.csv')
texts = df['Text'].astype(str).tolist()

print('Scoring 20 real positive resumes via V4.1 Engine...')
scores = calculate_match_scores(TEST_JOB, texts)

v4_verdicts = {}
for i, text in enumerate(texts):
    # Running 1-by-1 to preserve exact ordering
    res = calculate_match_scores(TEST_JOB, [text])[0]
    v4_verdicts[i] = res['score'] >= 38.0

print('Loading Gemini Subagent verdicts...')
gemini_verdicts = {}
try:
    with open('gemini_positives.json', 'r') as f:
        data = json.load(f)
        for item in data:
            abs_idx = item.get('index')
            gemini_verdicts[abs_idx] = item.get('gemini_pass', False)
except Exception as e:
    print(f'Error reading gemini_positives.json: {e}')

for i in range(len(texts)):
    if i not in gemini_verdicts:
        gemini_verdicts[i] = False

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for i in range(len(texts)):
    v4_pass = v4_verdicts[i]
    gemini_pass = gemini_verdicts[i]
    
    if v4_pass and gemini_pass:
        true_pos += 1
    elif v4_pass and not gemini_pass:
        false_pos += 1
    elif not v4_pass and not gemini_pass:
        true_neg += 1
    elif not v4_pass and gemini_pass:
        false_neg += 1

print('\n' + '='*50)
print('V4.1 ENGINE vs. GEMINI AI (POSITIVE GROUND TRUTH)')
print('='*50)
print(f'Total Validated: 20 Genuinely Real Python/Data Resumes')
print(f'True Positives (Both Passed): {true_pos}')
print(f'True Negatives (Both Failed): {true_neg}')
print(f'False Positives (V4 Passed, Gemini Failed): {false_pos}')
print(f'False Negatives (V4 Failed, Gemini Passed): {false_neg}')
print('='*50)

if (true_pos + false_neg) > 0:
    fnr = (false_neg / (true_pos + false_neg)) * 100
    print(f'FALSE NEGATIVE RATE relative to Gemini: {fnr:.2f}%')
else:
    print('FALSE NEGATIVE RATE relative to Gemini: N/A (Gemini rejected all)')

if (true_pos + false_neg) > 0:
    recall = (true_pos / (true_pos + false_neg)) * 100
    print(f'TARGET HIT RATE (Recall) relative to Gemini: {recall:.2f}%')

print('='*50)
