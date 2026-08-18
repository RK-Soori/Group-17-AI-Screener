import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# We will lazily load this so the server can start instantly.
sbert_model = None

def get_sbert():
    global sbert_model
    if sbert_model is None:
        print("Downloading/Loading Semantic Deep Learning Model (this may take a minute)...")
        try:
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model Loaded Successfully!")
        except Exception as e:
            print(f"Error loading SBERT: {e}")
    return sbert_model

# Domain-specific HR stopwords
HR_STOPWORDS = {'passionate', 'seeking', 'opportunity', 'team', 'player', 'responsibilities', 'duties', 'highly', 'motivated', 'driven', 'excellent', 'skills'}

# Important skills to weight heavier (we can duplicate them in text to artificially boost weight)
# I added a much larger list of tech skills to show how dynamic this is!
CORE_SKILLS = {
    'python', 'sql', 'aws', 'pandas', 'tensorflow', 'scikit-learn', 'machine learning', 
    'natural language processing', 'java', 'c++', 'react', 'node.js', 'docker', 
    'kubernetes', 'azure', 'git', 'linux', 'api'
}

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
    
    # 2. TF-IDF Lexical Math
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
        
        # 5. Hybrid Blended Score (The ultimate combo)
        # 50% Semantic Neural Net, 40% TF-IDF Lexical, 10% Jaccard Overlap
        blended_score = (cosine_sim_sbert[i] * 0.5) + (cosine_sim_tfidf[i] * 0.4) + (jac_score * 0.1)
        
        # Calculate percentage
        percentage = round(blended_score * 100, 2)
        
        # 6. Rule-Based Experience Bonus
        cand_exp = extract_years_experience(resumes[i])
        if job_exp_req > 0 and cand_exp >= job_exp_req:
            percentage += 15.0 # Flat 15% bonus
        
        # Cap at 100%
        percentage = min(percentage, 100.0)
        
        # Realistic Thresholds for Hybrid Model
        if percentage >= 60:
            result_label = "Highly Suitable"
        elif percentage >= 40:
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
