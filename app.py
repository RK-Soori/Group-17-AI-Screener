import os
import io
from flask import Flask, render_template, request, jsonify, session
from nlp_engine import calculate_match_scores

# PDF and DOCX text extraction
import PyPDF2
from docx import Document

app = Flask(__name__)
app.secret_key = 'ai-screener-group17-kdu-2026'  # Required for session storage

# Store last results in memory for the results page
last_results = {
    'results': [],
    'stats': {'highly_suitable': 0, 'suitable': 0, 'low_match': 0},
    'labels': [],
    'scores': []
}


# ==============================================================================
# TEXT EXTRACTION HELPERS
# ==============================================================================

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file stream."""
    reader = PyPDF2.PdfReader(file_stream)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return '\n'.join(text)


def extract_text_from_docx(file_stream):
    """Extract text from a DOCX file stream."""
    doc = Document(file_stream)
    text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)
    # Also extract from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text.append(cell.text)
    return '\n'.join(text)


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/')
def dashboard():
    """Dashboard / Home page with system overview and stats."""
    return render_template('dashboard.html', active_page='dashboard')


@app.route('/screen')
def screen():
    """Screening page with job description input and resume upload."""
    return render_template('screen.html', active_page='screen')


@app.route('/results')
def results():
    """Results page showing the last screening results."""
    return render_template('results.html',
                           active_page='results',
                           results=last_results['results'],
                           stats=last_results['stats'],
                           labels=last_results['labels'],
                           scores=last_results['scores'])


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload — extract text from PDF or DOCX and return it."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No file selected'}), 400

    filename = file.filename.lower()
    try:
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file.stream)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file.stream)
        else:
            return jsonify({'status': 'error', 'error': f'Unsupported file type: {filename}'}), 400

        if not text or not text.strip():
            return jsonify({'status': 'error', 'error': 'No text could be extracted from the file. The file may be image-based or empty.'}), 400

        return jsonify({'status': 'success', 'text': text, 'filename': file.filename})

    except Exception as e:
        return jsonify({'status': 'error', 'error': f'Failed to process file: {str(e)}'}), 500


@app.route('/score', methods=['POST'])
def score():
    """Run AI screening and return results as an HTMX partial or JSON."""
    global last_results

    data = request.get_json()
    job_desc = data.get('job_desc', '')
    resumes = data.get('resumes', [])

    if not job_desc or not resumes:
        return '<div class="text-center py-8 text-red-500 font-medium">Please provide a job description and at least one resume.</div>', 400

    # Run the AI matching engine
    results = calculate_match_scores(job_desc, resumes)

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)

    # Calculate stats
    stats = {
        'highly_suitable': sum(1 for r in results if r['label'] == 'Highly Suitable'),
        'suitable': sum(1 for r in results if r['label'] == 'Suitable'),
        'low_match': sum(1 for r in results if r['label'] == 'Low Match'),
    }

    labels = [r['candidate_id'] for r in results]
    scores = [r['score'] for r in results]

    # Store for the results page
    last_results = {
        'results': results,
        'stats': stats,
        'labels': labels,
        'scores': scores
    }

    # Check if HTMX request (returns HTML partial) or regular API call (returns JSON)
    if request.headers.get('HX-Request'):
        return render_template('partials/results_table.html',
                               results=results,
                               stats=stats,
                               labels=labels,
                               scores=scores)
    else:
        return jsonify(results)


@app.route('/feedback', methods=['POST'])
def feedback():
    """Capture human-in-the-loop feedback for future model retraining."""
    data = request.get_json()
    candidate_id = data.get('candidate_id')
    feedback_type = data.get('feedback')

    print(f"Human-in-the-loop Feedback Received: {candidate_id} -> {feedback_type}")

    return jsonify({"status": "success", "message": "Feedback recorded for future ML training!"})


# ==============================================================================
# LEGACY ROUTE (keep old index.html working)
# ==============================================================================

@app.route('/legacy')
def legacy():
    """Original V1 prototype interface (preserved for reference)."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
