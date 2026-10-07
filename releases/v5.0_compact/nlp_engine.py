import datetime
import re
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==============================================================================
# AI RESUME MATCHING ENGINE - COMPACT EDITION (V5.0)
# ==============================================================================
# Dynamic path resolver to locate model artifacts locally or from v4.1/web_app
# ==============================================================================

def resolve_asset_path(filename):
    """Resolve model asset path across current directory, v4.1, and web_app root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, filename),
        os.path.join(current_dir, "..", "v4.1", filename),
        os.path.join(current_dir, "..", "..", filename),
        os.path.join(current_dir, "..", filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(current_dir, filename)

TFIDF_MODEL_PATH = resolve_asset_path("domain_tfidf_model.pkl")
domain_tfidf_model = None
if os.path.exists(TFIDF_MODEL_PATH):
    try:
        domain_tfidf_model = joblib.load(TFIDF_MODEL_PATH)
        print(f"[v5.0_compact] Loaded Domain TF-IDF Model from {TFIDF_MODEL_PATH}")
    except Exception as e:
        print(f"[v5.0_compact] Warning: Could not load domain TF-IDF model: {e}")

sbert_model = None

def get_sbert():
    global sbert_model
    if sbert_model is None:
        fine_tuned_path = resolve_asset_path("final-hr-model")
        if os.path.exists(fine_tuned_path):
            print(f"[v5.0_compact] Loading Fine-Tuned Domain SBERT Model from {fine_tuned_path}...")
            try:
                sbert_model = SentenceTransformer(fine_tuned_path)
                print("[v5.0_compact] Fine-Tuned SBERT Loaded Successfully!")
            except Exception as e:
                print(f"[v5.0_compact] Fallback: {e}, using all-MiniLM-L6-v2")
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            print("[v5.0_compact] Loading Base Semantic Model (all-MiniLM-L6-v2)...")
            try:
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[v5.0_compact] Base Model Loaded Successfully!")
            except Exception as e:
                print(f"[v5.0_compact] Error loading SBERT: {e}")
    return sbert_model

# Domain HR stopwords
HR_STOPWORDS = {
    'passionate', 'seeking', 'opportunity', 'team', 'player', 'responsibilities', 
    'duties', 'highly', 'motivated', 'driven', 'excellent', 'skills'
}

# Expanded Semantic Skill Clusters (143+ Skills across 4 Core Engineering Domains)
SKILL_CLUSTERS = {
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

CORE_SKILLS = set().union(*SKILL_CLUSTERS.values())

TRANSFER_PAIRS = [
    # Data Analyst / BI -> AI / Machine Learning (+20.0%)
    (re.compile(r'\bdata\s*analyst\b|\bdata\s*analytics\b|\bbi\s*analyst\b', re.I),
     re.compile(r'\bai\b|\bmachine\s*learning\b|\bdata\s*science\b', re.I), 20.0),
    
    # Backend Engineer -> Frontend / Fullstack (+7.5%)
    (re.compile(r'\bbackend\b|\bjava\b|\bspring\b|\b\.net\b', re.I),
     re.compile(r'\bfrontend\b|\breact\b|\bweb\b', re.I), 7.5),

    # Sysadmin / IT Admin -> DevOps / Cloud (+4.0%)
    (re.compile(r'\bsysadmin\b|\bsystem\s*administrator\b|\blinux\s*admin\b', re.I),
     re.compile(r'\bdevops\b|\bcloud\b', re.I), 4.0),
]

SYNONYMS = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence'
}

def extract_years_experience(text):
    """Extract cumulative years of experience via mentions or date ranges."""
    text_l = text.lower()
    explicit = re.findall(r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)', text_l)
    valid_explicit = [int(x) for x in explicit if 1 <= int(x) <= 35]
    exp_explicit = max(valid_explicit) if valid_explicit else 0
    
    date_pat = re.compile(r'(\b(?:19[89]\d|20[012]\d)\b)\s*(?:-|to|–|—)\s*(\b(?:19[89]\d|20[012]\d)\b|present|current)', re.I)
    matches = date_pat.findall(text)
    exp_dates = 0
    if matches:
        CURRENT_YEAR = datetime.datetime.now().year
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

def preprocess_text_for_tfidf(text):
    """Clean text specifically for TF-IDF with domain token reinforcement."""
    text = text.lower()
    for key, val in SYNONYMS.items():
        text = re.sub(r'\b' + key + r'\b', val, text)
        
    for skill in CORE_SKILLS:
        if skill in text:
            text += f" {skill} {skill} {skill} {skill} {skill}"
            
    tokens = re.findall(r'\b\w+\b', text)
    clean_tokens = [t for t in tokens if t not in HR_STOPWORDS]
    return ' '.join(clean_tokens)

def jaccard_similarity(doc1, doc2):
    """Calculates Jaccard Similarity between token sets."""
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def chunk_text(text, max_words=140, overlap=35):
    """Sliding-window text chunker to overcome SBERT 256-token truncation."""
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

def calculate_match_scores(job_desc, resumes, model_version="v5"):
    """
    [v5.0 Compact] Production AI Ensemble Screening Engine
    Returns ranked candidate diagnostics with full explainability.
    """
    if not job_desc or not resumes:
        return []
    
    # Extract candidate IDs and resume texts robustly
    cand_ids = []
    resume_texts = []
    for idx, item in enumerate(resumes):
        if isinstance(item, dict):
            cid = item.get('name') or item.get('candidate_id') or item.get('id') or f"Candidate {idx + 1}"
            ctext = item.get('text') or item.get('content') or ''
        else:
            cid = f"Candidate {idx + 1}"
            ctext = str(item) if item is not None else ''
        cand_ids.append(str(cid).strip())
        resume_texts.append(ctext)

    # 1. Experience Requirements
    job_exp_req = extract_years_experience(job_desc)
    
    # 2. TF-IDF Lexical Processing
    processed_job = preprocess_text_for_tfidf(job_desc)
    processed_resumes = [preprocess_text_for_tfidf(r) for r in resume_texts]
    
    if domain_tfidf_model is not None:
        job_tfidf = domain_tfidf_model.transform([processed_job])
        resume_tfidf = domain_tfidf_model.transform(processed_resumes)
    else:
        documents = [processed_job] + processed_resumes
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(documents)
        job_tfidf = tfidf_matrix[0:1]
        resume_tfidf = tfidf_matrix[1:]

    cosine_sim_tfidf = cosine_similarity(job_tfidf, resume_tfidf).flatten()
    
    # 3. Fine-Tuned SBERT Semantic Processing with Sliding-Window Chunking
    sbert_scores = [0.0] * len(resume_texts)
    if model_version in ['v3', 'v5']:
        model = get_sbert()
        job_embedding = model.encode([job_desc])
        
        sbert_scores = []
        for r in resume_texts:
            if not r.strip():
                sbert_scores.append(0.0)
                continue
            r_chunks = chunk_text(r)
            if len(r_chunks) == 1:
                emb = model.encode([r])
                sbert_scores.append(float(cosine_similarity(job_embedding, emb)[0][0]))
            else:
                chunk_embs = model.encode(r_chunks)
                chunk_sims = cosine_similarity(job_embedding, chunk_embs)[0]
                top2 = sorted(chunk_sims, reverse=True)[:2]
                sim_max_pool = float(0.7 * top2[0] + 0.3 * np.mean(top2))
                sbert_scores.append(sim_max_pool)
            
    # 4. Job Domain Skills
    job_skills = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', job_desc.lower())}
    
    results = []
    for i in range(len(resume_texts)):
        jac_score = jaccard_similarity(processed_job, processed_resumes[i])
        cand_skills = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', resume_texts[i].lower())}
        
        # Bounded Skill Cluster Taxonomy Affinity
        skill_affinity = 0.0
        exact_matches = set()
        if job_skills:
            exact_matches = job_skills.intersection(cand_skills)
            cluster_credits = 0.0
            for cluster_name, cluster in SKILL_CLUSTERS.items():
                job_in_cl = [s for s in job_skills if s in cluster and s not in cand_skills]
                cand_in_cl = [s for s in cand_skills if s in cluster and s not in job_skills]
                cluster_credits += min(len(job_in_cl), len(cand_in_cl)) * 0.5
            skill_affinity = (len(exact_matches) * 1.0 + cluster_credits) / len(job_skills)
        
        cand_exp = extract_years_experience(resume_texts[i])
        transfer_boost = 0.0
        
        if model_version == 'v1':
            # V1: Pure TF-IDF Baseline
            percentage = float(cosine_sim_tfidf[i]) * 100.0
            
        elif model_version == 'v3':
            # V3: Early Hybrid (45% SBERT, 35% TF-IDF, 20% Taxonomy)
            raw_blended = (sbert_scores[i] * 0.45) + (cosine_sim_tfidf[i] * 0.35) + (skill_affinity * 0.20)
            BASELINE_FLOOR = 0.22
            calibrated_score = max(0.0, (raw_blended - BASELINE_FLOOR) / (1.0 - BASELINE_FLOOR))
            percentage = calibrated_score * 100.0
            if job_exp_req > 0 and cand_exp >= job_exp_req:
                percentage += 10.0
                
        else:
            # V5: Production Ensemble (65% SBERT, 20% TF-IDF, 15% Taxonomy)
            raw_blended = (sbert_scores[i] * 0.65) + (cosine_sim_tfidf[i] * 0.20) + (skill_affinity * 0.15)
            
            # Baseline Floor Rescaling
            BASELINE_FLOOR = 0.15
            calibrated_score = max(0.0, (raw_blended - BASELINE_FLOOR) / (1.0 - BASELINE_FLOOR))
            percentage = calibrated_score * 100.0
            
            # Dynamic Experience Bonus
            if job_exp_req > 0 and cand_exp >= job_exp_req:
                relevance_factor = min(1.0, (calibrated_score ** 0.5))
                percentage += 15.0 * relevance_factor
            
            # Cross-Domain Career Transferability Index
            for cand_pat, job_pat, boost_val in TRANSFER_PAIRS:
                if cand_pat.search(resume_texts[i]) and job_pat.search(job_desc):
                    percentage += boost_val
                    transfer_boost = boost_val
                    break
                    
            # Technical Gatekeeper
            if not cand_skills:
                percentage = 0.0
        
        percentage = round(float(min(percentage, 100.0)), 2)
        
        # Decision Tiers
        if percentage >= 60.0:
            result_label = "Highly Suitable"
        elif percentage >= 38.0:
            result_label = "Suitable"
        else:
            result_label = "Low Match"
            
        results.append({
            'candidate_id': cand_ids[i],
            'score': percentage,
            'label': result_label,
            'sbert_score': round(float(sbert_scores[i] * 100), 2),
            'tfidf_score': round(float(cosine_sim_tfidf[i] * 100), 2),
            'skill_affinity': round(float(skill_affinity * 100), 2),
            'matched_skills': sorted(list(exact_matches)) if job_skills else [],
            'missing_skills': sorted(list(job_skills - cand_skills)) if job_skills else [],
            'experience_years': cand_exp,
            'transfer_boost': transfer_boost
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results
