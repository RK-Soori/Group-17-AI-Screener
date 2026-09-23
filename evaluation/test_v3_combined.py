import os
import sys
import json
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add web_app to path
sys.path.insert(0, os.path.abspath('web_app'))
from nlp_engine import (
    calculate_match_scores as calculate_match_scores_v2,
    preprocess_text_for_tfidf, domain_tfidf_model, get_sbert, 
    jaccard_similarity, TRANSFER_PAIRS
)

print("=" * 80)
print("BENCHMARKING v2.2 vs v3.0 ON REAL RESUME DATASET (120 CANDIDATES)")
print("=" * 80)

# 1. EXPANDED SKILL TAXONOMY
EXPANDED_SKILL_CLUSTERS = {
    'ai_data': {
        'python', 'machine learning', 'data science', 'deep learning', 'pytorch', 
        'tensorflow', 'nlp', 'natural language processing', 'scikit-learn', 'pandas', 
        'sql', 'tableau', 'data analysis', 'fastapi', 'numpy', 'scipy', 'keras', 
        'opencv', 'matplotlib', 'seaborn', 'nltk', 'spacy', 'hugging face', 'huggingface', 
        'transformers', 'llm', 'bert', 'xgboost', 'lightgbm', 'pyspark', 'spark', 
        'hadoop', 'hive', 'power bi', 'bigquery', 'snowflake', 'statistics', 'r', 
        'data modeling', 'etl', 'data engineering', 'data warehouse', 'airflow'
    },
    'frontend': {
        'javascript', 'typescript', 'react', 'html', 'html5', 'css', 'css3', 'redux', 
        'vue', 'vue.js', 'angular', 'frontend', 'web development', 'next.js', 'nextjs', 
        'nuxt', 'tailwind', 'bootstrap', 'sass', 'scss', 'webpack', 'vite', 'jquery', 
        'graphql', 'responsive', 'dom', 'es6', 'jest', 'cypress', 'ui', 'ux'
    },
    'backend': {
        'java', 'spring', 'spring boot', '.net', 'c#', 'c++', 'microservices', 
        'sql', 'api', 'rest', 'restful', 'flask', 'fastapi', 'backend', 'django', 
        'node.js', 'nodejs', 'express', 'ruby', 'rails', 'php', 'laravel', 'golang', 
        'go', 'rust', 'asp.net', 'hibernate', 'jpa', 'soap', 'grpc', 'kafka', 
        'rabbitmq', 'celery', 'redis', 'postgresql', 'postgres', 'mysql', 'mongodb', 
        'cassandra', 'dynamodb', 'sqlite', 'oracle'
    },
    'devops_cloud': {
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'ci/cd', 'jenkins', 
        'terraform', 'linux', 'git', 'sysadmin', 'cloud', 'helm', 'ansible', 
        'gitlab', 'github actions', 'prometheus', 'grafana', 'elk', 'splunk', 
        'nginx', 'apache', 'bash', 'shell', 'powershell', 'ubuntu', 'centos', 'rhel'
    }
}
EXPANDED_CORE_SKILLS = set().union(*EXPANDED_SKILL_CLUSTERS.values())

# 2. ROBUST EXPERIENCE PARSER (Explicit + Multi-Span Dates)
def robust_experience_extractor(text):
    text_l = text.lower()
    explicit = re.findall(r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)', text_l)
    valid_explicit = [int(x) for x in explicit if 1 <= int(x) <= 35]
    exp_explicit = max(valid_explicit) if valid_explicit else 0
    
    date_pat = re.compile(r'(\b(?:19[89]\d|20[012]\d)\b)\s*(?:-|to|–|—)\s*(\b(?:19[89]\d|20[012]\d)\b|present|current)', re.I)
    matches = date_pat.findall(text)
    exp_dates = 0
    if matches:
        CURRENT_YEAR = 2024
        spans = []
        for s, e in matches:
            s_yr = int(s)
            e_yr = CURRENT_YEAR if e.lower() in ['present', 'current'] else int(e)
            if 1990 <= s_yr <= CURRENT_YEAR and s_yr <= e_yr <= CURRENT_YEAR:
                spans.append((s_yr, e_yr))
        if spans:
            min_yr = min(s for s, e in spans)
            max_yr = max(e for s, e in spans)
            exp_dates = min(30, max_yr - min_yr)
            
    return max(exp_explicit, exp_dates)

# 3. TEXT CHUNKING FOR SBERT
def chunk_text(text, max_words=140, overlap=35):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return chunks

model = get_sbert()

def score_candidate_v3(job_desc, cv_text):
    explicit_job_exp = re.findall(r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)', job_desc.lower())
    job_exp_req = max([int(x) for x in explicit_job_exp if 1 <= int(x) <= 30], default=0)
    
    p_job = preprocess_text_for_tfidf(job_desc)
    p_cv = preprocess_text_for_tfidf(cv_text)
    jt = domain_tfidf_model.transform([p_job])
    ct = domain_tfidf_model.transform([p_cv])
    sim_tfidf = float(cosine_similarity(jt, ct)[0][0])
    
    je = model.encode([job_desc])
    cv_chunks = chunk_text(cv_text)
    if len(cv_chunks) == 1:
        ce = model.encode([cv_text])
        sim_sbert = float(cosine_similarity(je, ce)[0][0])
    else:
        chunk_embs = model.encode(cv_chunks)
        chunk_sims = cosine_similarity(je, chunk_embs)[0]
        top2 = sorted(chunk_sims, reverse=True)[:2]
        sim_sbert = float(0.7 * top2[0] + 0.3 * np.mean(top2))
        
    jac = jaccard_similarity(p_job, p_cv)
    
    job_skills = {s for s in EXPANDED_CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', job_desc.lower())}
    cand_skills = {s for s in EXPANDED_CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', cv_text.lower())}
    
    exact_matches = job_skills.intersection(cand_skills)
    cluster_credits = 0.0
    for cl_name, cl in EXPANDED_SKILL_CLUSTERS.items():
        job_in_cl = [s for s in job_skills if s in cl and s not in cand_skills]
        cand_in_cl = [s for s in cand_skills if s in cl and s not in job_skills]
        cluster_credits += min(len(job_in_cl), len(cand_in_cl)) * 0.5
        
    skill_affinity = (len(exact_matches) * 1.0 + cluster_credits) / max(1, len(job_skills))
    
    raw_blended = (sim_sbert * 0.45) + (sim_tfidf * 0.35) + (skill_affinity * 0.15) + (jac * 0.05)
    calibrated = max(0.0, (raw_blended - 0.22) / 0.78)
    percentage = calibrated * 100.0
    
    cand_exp = robust_experience_extractor(cv_text)
    if job_exp_req > 0 and cand_exp >= job_exp_req:
        percentage += 15.0 * min(1.0, (calibrated ** 0.5))
        
# 4. v3 CAREER TRANSFERABILITY PAIRS
V3_TRANSFER_PAIRS = [
    # Data Analyst / BI -> AI / Machine Learning (+20.0%)
    (re.compile(r'\bdata\s*analyst\b|\bdata\s*analytics\b|\bbi\s*analyst\b', re.I),
     re.compile(r'\bai\b|\bmachine\s*learning\b|\bdata\s*science\b', re.I), 20.0),
    # Backend Engineer -> Frontend / Fullstack (+7.5% - transfer programming foundations & APIs)
    (re.compile(r'\bbackend\b|\bjava\b|\bspring\b|\b\.net\b', re.I),
     re.compile(r'\bfrontend\b|\breact\b|\bweb\b', re.I), 7.5),
    # Sysadmin / IT Admin -> DevOps / Cloud (+4.0% - Linux, Scripting, Network administration)
    (re.compile(r'\bsysadmin\b|\bsystem\s*administrator\b|\blinux\s*admin\b', re.I),
     re.compile(r'\bdevops\b|\bcloud\b', re.I), 4.0),
]

def score_candidate_v3(job_desc, cv_text):
    explicit_job_exp = re.findall(r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)', job_desc.lower())
    job_exp_req = max([int(x) for x in explicit_job_exp if 1 <= int(x) <= 30], default=0)
    
    p_job = preprocess_text_for_tfidf(job_desc)
    p_cv = preprocess_text_for_tfidf(cv_text)
    jt = domain_tfidf_model.transform([p_job])
    ct = domain_tfidf_model.transform([p_cv])
    sim_tfidf = float(cosine_similarity(jt, ct)[0][0])
    
    je = model.encode([job_desc])
    cv_chunks = chunk_text(cv_text)
    if len(cv_chunks) == 1:
        ce = model.encode([cv_text])
        sim_sbert = float(cosine_similarity(je, ce)[0][0])
    else:
        chunk_embs = model.encode(cv_chunks)
        chunk_sims = cosine_similarity(je, chunk_embs)[0]
        top2 = sorted(chunk_sims, reverse=True)[:2]
        sim_sbert = float(0.7 * top2[0] + 0.3 * np.mean(top2))
        
    jac = jaccard_similarity(p_job, p_cv)
    
    job_skills = {s for s in EXPANDED_CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', job_desc.lower())}
    cand_skills = {s for s in EXPANDED_CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', cv_text.lower())}
    
    exact_matches = job_skills.intersection(cand_skills)
    cluster_credits = 0.0
    for cl_name, cl in EXPANDED_SKILL_CLUSTERS.items():
        job_in_cl = [s for s in job_skills if s in cl and s not in cand_skills]
        cand_in_cl = [s for s in cand_skills if s in cl and s not in job_skills]
        cluster_credits += min(len(job_in_cl), len(cand_in_cl)) * 0.5
        
    skill_affinity = (len(exact_matches) * 1.0 + cluster_credits) / max(1, len(job_skills))
    
    raw_blended = (sim_sbert * 0.45) + (sim_tfidf * 0.35) + (skill_affinity * 0.15) + (jac * 0.05)
    calibrated = max(0.0, (raw_blended - 0.22) / 0.78)
    percentage = calibrated * 100.0
    
    cand_exp = robust_experience_extractor(cv_text)
    if job_exp_req > 0 and cand_exp >= job_exp_req:
        percentage += 15.0 * min(1.0, (calibrated ** 0.5))
        
    for cand_pat, job_pat, boost_val in V3_TRANSFER_PAIRS:
        if cand_pat.search(cv_text) and job_pat.search(job_desc):
            percentage += boost_val
            break
            
    # Technical Gatekeeper: Zero technical skills detected => Hard 0.0%
    if not cand_skills:
        percentage = 0.0
        
    return round(min(100.0, percentage), 2)

def run_real_benchmark():
    # Load real dataset
    df = pd.read_csv('Resume_Dataset_Real.csv')

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

    v2_target_scores = []
    v3_target_scores = []
    v2_adj_scores = []
    v3_adj_scores = []
    v2_irrel_scores = []
    v3_irrel_scores = []

    results_by_cat_v2 = {}
    results_by_cat_v3 = {}

    print("\nRunning Evaluation across 120 Candidates...")

    for job in JOBS:
        job_desc = job['description']
        print(f"\nProcessing Job: {job['role']}...")
    
        for tier, categories in job['test_categories'].items():
            for cat in categories:
                resumes = df[df['Category'] == cat].head(10)['Text'].tolist()
                
                if cat not in results_by_cat_v2:
                    results_by_cat_v2[cat] = []
                    results_by_cat_v3[cat] = []
                    
                for cv in resumes:
                    s2 = calculate_match_scores_v2(job_desc, [cv])[0]['score']
                    s3 = score_candidate_v3(job_desc, cv)
                    
                    results_by_cat_v2[cat].append(s2)
                    results_by_cat_v3[cat].append(s3)
                    
                    if "Target" in tier:
                        v2_target_scores.append(s2)
                        v3_target_scores.append(s3)
                    elif "Adjacent" in tier:
                        v2_adj_scores.append(s2)
                        v3_adj_scores.append(s3)
                    else:
                        v2_irrel_scores.append(s2)
                        v3_irrel_scores.append(s3)

    print("\n" + "=" * 80)
    print("REAL RESUME BENCHMARK AGGREGATE SUMMARY: v2.2 vs v3.0")
    print("=" * 80)
    print(f"Target Technical Mean:     v2.2 = {np.mean(v2_target_scores):.2f}%  -->  v3.0 = {np.mean(v3_target_scores):.2f}%  (Improvement: +{np.mean(v3_target_scores) - np.mean(v2_target_scores):.2f}%)")
    print(f"Adjacent Cross-Domain Mean: v2.2 = {np.mean(v2_adj_scores):.2f}%  -->  v3.0 = {np.mean(v3_adj_scores):.2f}%  (Improvement: +{np.mean(v3_adj_scores) - np.mean(v2_adj_scores):.2f}%)")
    print(f"Irrelevant Non-Tech Mean:   v2.2 = {np.mean(v2_irrel_scores):.2f}%  -->  v3.0 = {np.mean(v3_irrel_scores):.2f}%")
    print(f"Target Hit Rate (>=45%):    v2.2 = {(np.array(v2_target_scores) >= 45).mean()*100:.1f}%  -->  v3.0 = {(np.array(v3_target_scores) >= 45).mean()*100:.1f}%")
    print(f"Non-Tech FPR (>=45%):       v2.2 = {(np.array(v2_irrel_scores) >= 45).mean()*100:.2f}%  -->  v3.0 = {(np.array(v3_irrel_scores) >= 45).mean()*100:.2f}%")
    print(f"Discrimination Gap:         v2.2 = +{np.mean(v2_target_scores) - np.mean(v2_irrel_scores):.2f}%  -->  v3.0 = +{np.mean(v3_target_scores) - np.mean(v3_irrel_scores):.2f}%")
    print("=" * 80)

    print("\nDetailed Per-Category Average Comparison:")
    print(f"{'Category':<26} | {'v2.2 Mean':<10} | {'v3.0 Mean':<10} | {'Delta':<10}")
    print("-" * 62)
    for cat in results_by_cat_v2:
        m2 = np.mean(results_by_cat_v2[cat])
        m3 = np.mean(results_by_cat_v3[cat])
        print(f"{cat:<26} | {m2:6.2f}%    | {m3:6.2f}%    | {m3 - m2:+6.2f}%")

    # Save results to json
    summary_data = {
        "v2": {
            "target_mean": float(np.mean(v2_target_scores)),
            "adjacent_mean": float(np.mean(v2_adj_scores)),
            "irrelevant_mean": float(np.mean(v2_irrel_scores)),
            "target_hit_rate": float((np.array(v2_target_scores) >= 45).mean() * 100),
            "non_tech_fpr": float((np.array(v2_irrel_scores) >= 45).mean() * 100)
        },
        "v3": {
            "target_mean": float(np.mean(v3_target_scores)),
            "adjacent_mean": float(np.mean(v3_adj_scores)),
            "irrelevant_mean": float(np.mean(v3_irrel_scores)),
            "target_hit_rate": float((np.array(v3_target_scores) >= 45).mean() * 100),
            "non_tech_fpr": float((np.array(v3_irrel_scores) >= 45).mean() * 100)
        }
    }

    with open('v3_benchmark_results.json', 'w') as f:
        json.dump(summary_data, f, indent=4)
    print("\nSaved benchmark results to v3_benchmark_results.json")

if __name__ == '__main__':
    run_real_benchmark()
