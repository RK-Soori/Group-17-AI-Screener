import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load pre-fitted Domain TF-IDF model if available
TFIDF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "domain_tfidf_model.pkl")
domain_tfidf_model = None
if os.path.exists(TFIDF_MODEL_PATH):
    try:
        domain_tfidf_model = joblib.load(TFIDF_MODEL_PATH)
        print(f"Loaded Domain TF-IDF Model from {TFIDF_MODEL_PATH}")
    except Exception as e:
        print(f"Could not load domain TF-IDF model: {e}")

# We will lazily load this so the server can start instantly.
sbert_model = None

def get_sbert():
    global sbert_model
    if sbert_model is None:
        fine_tuned_path = os.path.join(os.path.dirname(__file__), "final-hr-model")
        if os.path.exists(fine_tuned_path):
            print(f"Loading Fine-Tuned Domain SBERT Model from {fine_tuned_path}...")
            try:
                sbert_model = SentenceTransformer(fine_tuned_path)
                print("Fine-Tuned SBERT Loaded Successfully!")
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}, falling back to base model.")
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            print("Downloading/Loading Base Semantic Model (all-MiniLM-L6-v2)...")
            try:
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Base Model Loaded Successfully!")
            except Exception as e:
                print(f"Error loading SBERT: {e}")
    return sbert_model

# Domain-specific HR stopwords
HR_STOPWORDS = {'passionate', 'seeking', 'opportunity', 'team', 'player', 'responsibilities', 'duties', 'highly', 'motivated', 'driven', 'excellent', 'skills'}

# Semantic Skill Clusters for partial domain credit & skill taxonomy
SKILL_CLUSTERS = {
    'ai_data': {
        'python', 'machine learning', 'data science', 'deep learning', 'pytorch', 
        'tensorflow', 'nlp', 'natural language processing', 'scikit-learn', 'pandas', 
        'sql', 'tableau', 'data analysis', 'fastapi'
    },
    'frontend': {
        'javascript', 'typescript', 'react', 'html', 'css', 'redux', 'vue', 
        'angular', 'frontend', 'ui', 'ux', 'web development'
    },
    'backend': {
        'java', 'spring', 'spring boot', '.net', 'c#', 'c++', 'microservices', 
        'sql', 'api', 'rest', 'flask', 'fastapi', 'backend', 'django'
    },
    'devops_cloud': {
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'jenkins', 
        'terraform', 'linux', 'git', 'sysadmin', 'cloud'
    }
}

# The complete universe of recognized technical competencies
CORE_SKILLS = set().union(*SKILL_CLUSTERS.values())

# [v2.2] Cross-Domain Career Transferability Matrix
# Defines realistic domain adjacency weights between candidate background and job targets
TRANSFER_PAIRS = [
    # Data Analyst / BI -> AI / Machine Learning (Data foundations, Python, Stats, ML basics)
    (re.compile(r'\bdata\s*analyst\b|\bdata\s*analytics\b|\bbi\s*analyst\b', re.I),
     re.compile(r'\bai\b|\bmachine\s*learning\b|\bdata\s*science\b', re.I), 20.0),
    
    # Backend Engineer -> Frontend / Fullstack (Shared programming foundations, APIs, Databases)
    (re.compile(r'\bbackend\b|\bjava\b|\bspring\b|\b\.net\b', re.I),
     re.compile(r'\bfrontend\b|\breact\b|\bweb\b', re.I), 6.0),

    # Sysadmin / IT Admin -> DevOps / Cloud (Linux, Scripting, Network administration)
    (re.compile(r'\bsysadmin\b|\bsystem\s*administrator\b|\blinux\s*admin\b', re.I),
     re.compile(r'\bdevops\b|\bcloud\b', re.I), 4.0),
]

SYNONYMS = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence'
}

def extract_years_experience(text):
    """Rule-Based matcher for experience"""
    match = re.search(r'(\d+)\s*(?:\+)?\s*years?', text.lower())
    if match:
        return int(match.group(1))
    return 0

def preprocess_text_for_tfidf(text):
    """Clean text specifically for TF-IDF (removing HR fluff)"""
    text = text.lower()
    
    # Semantic mapping (synonyms)
    for key, val in SYNONYMS.items():
        text = re.sub(r'\b' + key + r'\b', val, text)
        
    # ** THE Keyword Weighting Strategy **
    for skill in CORE_SKILLS:
        if skill in text:
            # Add the skill 5 EXTRA TIMES to massively boost its mathematical weight!
            text += f" {skill} {skill} {skill} {skill} {skill}"
            
    tokens = re.findall(r'\b\w+\b', text)
    clean_tokens = [t for t in tokens if t not in HR_STOPWORDS]
    return ' '.join(clean_tokens)

def jaccard_similarity(doc1, doc2):
    """Calculates Jaccard Similarity (intersection over union)"""
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def calculate_match_scores(job_desc, resumes):
    if not job_desc or not resumes:
        return []
    
    # 1. Rule-Based setup
    job_exp_req = extract_years_experience(job_desc)
    
    # Preprocess all documents for TF-IDF (Lexical match)
    processed_job = preprocess_text_for_tfidf(job_desc)
    processed_resumes = [preprocess_text_for_tfidf(r) for r in resumes]
    documents = [processed_job] + processed_resumes
    
    # 2. TF-IDF Lexical Math (Domain-Adapted)
    if domain_tfidf_model is not None:
        job_tfidf = domain_tfidf_model.transform([processed_job])
        resume_tfidf = domain_tfidf_model.transform(processed_resumes)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(documents)
        job_tfidf = tfidf_matrix[0:1]
        resume_tfidf = tfidf_matrix[1:]

    cosine_sim_tfidf = cosine_similarity(job_tfidf, resume_tfidf).flatten()
    
    # 3. Semantic Deep Learning Math (SBERT)
    model = get_sbert()
    if model:
        # We give the neural network the RAW text so it understands full sentence context
        job_embedding = model.encode([job_desc])
        resume_embeddings = model.encode(resumes)
        cosine_sim_sbert = cosine_similarity(job_embedding, resume_embeddings).flatten()
    else:
        # Fallback if model fails to load
        cosine_sim_sbert = [0] * len(resumes)
    
    results = []
    for i in range(len(resumes)):
        # 4. Calculate Jaccard
        jac_score = jaccard_similarity(processed_job, processed_resumes[i])
        
        # [v2.2] Bounded Skill Cluster Taxonomy Affinity
        # Partial credit is proportional to the candidate's actual competencies within the cluster
        job_skills = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', job_desc.lower())}
        cand_skills = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', resumes[i].lower())}
        
        skill_affinity = 0.0
        if job_skills:
            exact_matches = job_skills.intersection(cand_skills)
            cluster_credits = 0.0
            for cluster_name, cluster in SKILL_CLUSTERS.items():
                job_in_cl = [s for s in job_skills if s in cluster and s not in cand_skills]
                cand_in_cl = [s for s in cand_skills if s in cluster and s not in job_skills]
                cluster_credits += min(len(job_in_cl), len(cand_in_cl)) * 0.5
            skill_affinity = (len(exact_matches) * 1.0 + cluster_credits) / len(job_skills)
        
        # 5. [v2.0] Hybrid Blended Score (45% SBERT + 35% TF-IDF + 15% Skill Cluster Taxonomy + 5% Jaccard)
        raw_blended = (cosine_sim_sbert[i] * 0.45) + (cosine_sim_tfidf[i] * 0.35) + (skill_affinity * 0.15) + (jac_score * 0.05)
        
        # [v1.1] Baseline Floor Rescaling (Removes universal ~0.22 English cosine floor)
        BASELINE_FLOOR = 0.22
        calibrated_score = max(0.0, (raw_blended - BASELINE_FLOOR) / (1.0 - BASELINE_FLOOR))
        percentage = round(calibrated_score * 100, 2)
        
        # 6. [v2.1] Rule-Based Dynamic Experience Bonus (Square-Root Relevance Scaling)
        # Uses sqrt(calibrated_score) so partial technical matches receive appropriate credit
        # while non-technical candidates (score 0.0) still receive exactly 0.0 bonus!
        cand_exp = extract_years_experience(resumes[i])
        if job_exp_req > 0 and cand_exp >= job_exp_req:
            relevance_factor = min(1.0, (calibrated_score ** 0.5))
            dynamic_bonus = 15.0 * relevance_factor
            percentage += dynamic_bonus
        
        # 7. [v2.2] Cross-Domain Career Transferability Index
        # Boosts verified mid-tier candidates transitioning from adjacent technical disciplines
        transfer_boost = 0.0
        for cand_pat, job_pat, boost_val in TRANSFER_PAIRS:
            if cand_pat.search(resumes[i]) and job_pat.search(job_desc):
                transfer_boost = max(transfer_boost, boost_val)
                break
        percentage += transfer_boost

        # 8. [v1.4] Calibrated Technical Gatekeeper (Non-Technical Profile Elimination)
        cand_all_tech = {s for s in CORE_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', resumes[i].lower())}
        if not cand_all_tech:
            percentage = 0.0
        
        percentage = round(min(percentage, 100.0), 2)
        
        # [v2.1] Recalibrated Decision Thresholds
        # 65% for Highly Suitable, 45% for Suitable, <45% for Low Match
        if percentage >= 65:
            result_label = "Highly Suitable"
        elif percentage >= 45:
            result_label = "Suitable"
        else:
            result_label = "Low Match"
            
        results.append({
            'candidate_id': f"Candidate {i + 1}",
            'score': percentage,
            'label': result_label
        })
    
    # Sort results from highest score to lowest
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results
