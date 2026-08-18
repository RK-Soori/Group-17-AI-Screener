from flask import Flask, render_template, request, jsonify
from nlp_engine import calculate_match_scores

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    data = request.get_json()
    job_desc = data.get('job_desc', '')
    resumes = data.get('resumes', [])
    
    # Process the text using our advanced NLP logic
    results = calculate_match_scores(job_desc, resumes)
    
    return jsonify(results)

@app.route('/feedback', methods=['POST'])
def feedback():
    """Mock endpoint to capture human-in-the-loop feedback"""
    data = request.get_json()
    candidate_id = data.get('candidate_id')
    feedback_type = data.get('feedback')
    
    print(f"Human-in-the-loop Feedback Received: {candidate_id} -> {feedback_type}")
    
    # In a real app, this would save to a database to retrain the model weights later
    return jsonify({"status": "success", "message": "Feedback recorded for future ML training!"})

if __name__ == '__main__':
    app.run(debug=True)
