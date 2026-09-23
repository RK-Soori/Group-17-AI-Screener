import sys
import os
import json
import numpy as np

# Add web_app to path to import nlp_engine
sys.path.insert(0, os.path.abspath('web_app'))
from nlp_engine import calculate_match_scores

# ---------------------------------------------------------------------------
# BENCHMARK TEST SUITE: 3 Real-World Jobs & 12 Diverse Candidate Resumes
# ---------------------------------------------------------------------------
TEST_SUITE = [
    {
        "job_id": "JOB-1",
        "role": "Senior Python & AI Engineer",
        "description": "Looking for a Senior Python & AI Engineer with at least 3 years of experience. Must be proficient in Python, PyTorch or TensorFlow, Natural Language Processing, building REST APIs with FastAPI or Flask, and containerization using Docker and Linux.",
        "candidates": [
            {
                "id": "Cand-1A",
                "name": "Alex Chen (Senior AI Engineer)",
                "resume": "Senior AI & Machine Learning Engineer with 4 years of experience. Proficient in Python, PyTorch, TensorFlow, Natural Language Processing (NLP), transformers, Hugging Face, Flask, FastAPI, Docker, and Linux. Built and deployed scalable LLM applications.",
                "gemini_expected_score": 90.0,
                "gemini_expected_label": "Highly Suitable"
            },
            {
                "id": "Cand-1B",
                "name": "Sarah Miller (Data Analyst)",
                "resume": "Data Analyst with 3 years experience. Skilled in Python, SQL, Pandas, Tableau, and basic Machine Learning algorithms like Scikit-Learn regression and classification. Experienced in data visualization and reporting.",
                "gemini_expected_score": 55.0,
                "gemini_expected_label": "Suitable"
            },
            {
                "id": "Cand-1C",
                "name": "David Clark (Junior Web Developer)",
                "resume": "Junior Web Developer with 1 year experience in HTML, CSS, JavaScript, and basic Python scripting. Passionate about AI and eager to learn machine learning frameworks.",
                "gemini_expected_score": 30.0,
                "gemini_expected_label": "Low Match"
            },
            {
                "id": "Cand-1D",
                "name": "Marcus Vance (Executive Chef)",
                "resume": "Head Chef with 8 years of culinary and kitchen management experience. Expert in menu development, food safety protocols, inventory control, and staff supervision.",
                "gemini_expected_score": 5.0,
                "gemini_expected_label": "Low Match"
            }
        ]
    },
    {
        "job_id": "JOB-2",
        "role": "Frontend React Developer",
        "description": "Hiring a Frontend Developer with 2+ years experience building responsive web apps using React, JavaScript, TypeScript, HTML5, CSS3, and Redux. Experience with Git and REST APIs required.",
        "candidates": [
            {
                "id": "Cand-2A",
                "name": "Elena Rostova (React Developer)",
                "resume": "Frontend Developer with 3 years of experience specializing in React, TypeScript, JavaScript, Redux, HTML5, modern CSS, Git, and integrating RESTful APIs.",
                "gemini_expected_score": 88.0,
                "gemini_expected_label": "Highly Suitable"
            },
            {
                "id": "Cand-2B",
                "name": "Kevin Patel (Backend Java Dev)",
                "resume": "Backend Software Engineer with 4 years experience in Java, Spring Boot, Microservices, and SQL. Has basic working knowledge of JavaScript and HTML for internal admin dashboards.",
                "gemini_expected_score": 48.0,
                "gemini_expected_label": "Suitable"
            },
            {
                "id": "Cand-2C",
                "name": "Lisa Wong (UI/UX Designer)",
                "resume": "UI/UX Designer with 2 years experience creating wireframes, Figma mockups, and user research. Limited coding experience with basic HTML and CSS.",
                "gemini_expected_score": 32.0,
                "gemini_expected_label": "Low Match"
            },
            {
                "id": "Cand-2D",
                "name": "Robert Taylor (Chartered Accountant)",
                "resume": "Certified Public Accountant with 6 years experience in tax filing, auditing, financial statements, balance sheets, and QuickBooks.",
                "gemini_expected_score": 5.0,
                "gemini_expected_label": "Low Match"
            }
        ]
    },
    {
        "job_id": "JOB-3",
        "role": "DevOps & Cloud Engineer",
        "description": "Seeking a DevOps & Cloud Engineer with 4 years experience. Key requirements include AWS, Kubernetes, Docker, CI/CD automation with Jenkins/GitLab, Terraform, and Linux administration.",
        "candidates": [
            {
                "id": "Cand-3A",
                "name": "Tariq Mansoor (Senior DevOps)",
                "resume": "Lead DevOps Engineer with 5 years experience managing AWS infrastructure, Kubernetes clusters, Docker containers, CI/CD pipelines with GitLab, Terraform, and Linux systems.",
                "gemini_expected_score": 92.0,
                "gemini_expected_label": "Highly Suitable"
            },
            {
                "id": "Cand-3B",
                "name": "Brian Adams (Linux Sysadmin)",
                "resume": "System Administrator with 4 years experience in Linux server management, bash scripting, network troubleshooting, and introductory Docker experience. Actively studying for AWS solutions architect.",
                "gemini_expected_score": 52.0,
                "gemini_expected_label": "Suitable"
            },
            {
                "id": "Cand-3C",
                "name": "Emily Watson (Technical Support)",
                "resume": "IT Support Specialist with 2 years experience in desktop support, user account provisioning, Windows/Linux troubleshooting, and ticketing systems.",
                "gemini_expected_score": 28.0,
                "gemini_expected_label": "Low Match"
            },
            {
                "id": "Cand-3D",
                "name": "James Sullivan (Civil Project Manager)",
                "resume": "Civil Construction Manager with 7 years experience overseeing structural projects, contractor safety, concrete inspections, and CAD blueprints.",
                "gemini_expected_score": 5.0,
                "gemini_expected_label": "Low Match"
            }
        ]
    }
]

def run_benchmark():
    total_pairs = 0
    correct_labels = 0
    correct_rank_1 = 0
    absolute_errors = []
    
    detailed_results = []
    
    print("=" * 80)
    print("RUNNING BENCHMARK EVALUATION: YOUR TRAINED SYSTEM vs. GEMINI GROUND TRUTH")
    print("=" * 80)

    for test_case in TEST_SUITE:
        job_role = test_case["role"]
        job_desc = test_case["description"]
        candidates = test_case["candidates"]
        
        # Prepare inputs for your system
        raw_resumes = [c["resume"] for c in candidates]
        
        # Run local system matching
        # Note: calculate_match_scores sorts them, so let's match them back by candidate text
        system_outputs = calculate_match_scores(job_desc, raw_resumes)
        
        # Pair outputs back to the candidate IDs
        # To preserve exact candidate mapping, test each candidate individually or map via score
        # Let's map individually to ensure exact 1-to-1 candidate accuracy
        cand_results = []
        for cand in candidates:
            out = calculate_match_scores(job_desc, [cand["resume"]])[0]
            cand_results.append({
                "cand_id": cand["id"],
                "cand_name": cand["name"],
                "system_score": float(out["score"]),
                "system_label": out["label"],
                "gemini_score": cand["gemini_expected_score"],
                "gemini_label": cand["gemini_expected_label"]
            })
            
            # Metrics computation
            total_pairs += 1
            if out["label"] == cand["gemini_expected_label"]:
                correct_labels += 1
            absolute_errors.append(abs(float(out["score"]) - cand["gemini_expected_score"]))

        # Check if the top-ranked candidate by system is indeed Candidate A (the intended top match)
        cand_results_sorted = sorted(cand_results, key=lambda x: x["system_score"], reverse=True)
        if cand_results_sorted[0]["cand_id"].endswith("A"):
            correct_rank_1 += 1

        detailed_results.append({
            "job_role": job_role,
            "results": cand_results
        })

    # Summary Statistics
    label_accuracy = (correct_labels / total_pairs) * 100
    top_1_accuracy = (correct_rank_1 / len(TEST_SUITE)) * 100
    mae = np.mean(absolute_errors)
    
    # Pearson Correlation between System and Gemini Scores
    all_sys_scores = [r["system_score"] for jr in detailed_results for r in jr["results"]]
    all_gem_scores = [r["gemini_score"] for jr in detailed_results for r in jr["results"]]
    correlation = np.corrcoef(all_sys_scores, all_gem_scores)[0, 1]

    # Print results to stdout
    print(f"\nTotal Test Pairs Evaluated: {total_pairs}")
    print(f"Classification Label Accuracy: {label_accuracy:.2f}% ({correct_labels}/{total_pairs} exact category matches)")
    print(f"Top-1 Candidate Ranking Accuracy (Precision@1): {top_1_accuracy:.2f}% ({correct_rank_1}/{len(TEST_SUITE)} correct #1 picks)")
    print(f"Mean Absolute Error (MAE): {mae:.2f}%")
    print(f"Pearson Correlation (System vs Gemini): {correlation:.4f}")
    print("=" * 80)
    
    # Save output to JSON for easy viewing
    report = {
        "summary": {
            "total_pairs": total_pairs,
            "label_accuracy_percent": round(label_accuracy, 2),
            "top_1_ranking_accuracy_percent": round(top_1_accuracy, 2),
            "mean_absolute_error": round(float(mae), 2),
            "pearson_correlation": round(float(correlation), 4)
        },
        "breakdown": detailed_results
    }
    
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print("Benchmark results saved to 'benchmark_results.json'.")

if __name__ == "__main__":
    run_benchmark()
