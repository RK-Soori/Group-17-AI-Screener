import re

with open('nlp_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('VERSION 3.0', 'VERSION 5.0')
content = content.replace('[v3.0]', '[v5.0]')
content = content.replace('0.22 floor removal', '0.15 floor removal')

if 'import datetime' not in content:
    content = 'import datetime\n' + content

content = re.sub(r'CURRENT_YEAR = \d{4}', 'CURRENT_YEAR = datetime.datetime.now().year', content)

old_append = '''        results.append({
            'candidate_id': f"Candidate {i + 1}",
            'score': percentage,
            'label': result_label
        })'''
new_append = '''        results.append({
            'candidate_id': f"Candidate {i + 1}",
            'score': percentage,
            'label': result_label,
            'sbert_score': round(sbert_scores[i] * 100, 2),
            'tfidf_score': round(cosine_sim_tfidf[i] * 100, 2),
            'skill_affinity': round(skill_affinity * 100, 2),
            'matched_skills': list(exact_matches) if job_skills else [],
            'missing_skills': list(job_skills - cand_skills) if job_skills else [],
            'experience_years': cand_exp
        })'''
content = content.replace(old_append, new_append)

with open('nlp_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
