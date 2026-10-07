import unittest
import json
import os
import io
from app import app, BASE_DIR

class TestCompactUI(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_routes_health_and_index(self):
        """Test GET / and /status routes."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn('Candidate Screening Cockpit', html)
        self.assertIn('Ensemble Architecture & Benchmarks', html)
        self.assertIn('Ranked Candidate Telemetry', html)

        # Test /status
        res_status = self.client.get('/status')
        self.assertEqual(res_status.status_code, 200)
        data = json.loads(res_status.data)
        self.assertEqual(data.get('status'), 'healthy')
        self.assertEqual(data.get('engine'), 'v5.0_compact')

    def test_redirects(self):
        """Test navigation redirects to anchors."""
        for path, target in [('/screen', '/#screen'), ('/results', '/#results'), ('/architecture', '/#architecture')]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 302)
            self.assertEqual(res.headers.get('Location'), target)

    def test_api_presets(self):
        """Test preset loader returns sample requisitions."""
        res = self.client.get('/api/presets')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('ai_engineer', data)
        self.assertIn('frontend_react', data)
        self.assertIn('devops_cloud', data)
        self.assertGreater(len(data['ai_engineer']['candidates']), 0)

    def test_scoring_api_json(self):
        """Test POST /score returning JSON."""
        payload = {
            "job_desc": "Senior Python and AI Engineer with experience in PyTorch, transformers, and NLP.",
            "resumes": [
                "Senior AI Engineer with 4 years experience in Python, PyTorch, transformers, and NLP.",
                "Executive chef with culinary experience in kitchen management and recipes."
            ],
            "model_version": "v5"
        }
        res = self.client.post('/score', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(len(data.get('results')), 2)
        
        # Candidate 1 should be Highly Suitable, Candidate 2 Low Match
        cand1 = data['results'][0]
        cand2 = data['results'][1]
        self.assertGreater(cand1['score'], cand2['score'])
        self.assertEqual(cand2['score'], 0.0) # Zero-match gatekeeper

    def test_scoring_htmx_partial(self):
        """Test POST /score with HX-Request returning HTML partial."""
        payload = {
            "job_desc": "Frontend Developer with React, TypeScript, HTML5, CSS3, and Redux experience.",
            "resumes": [
                "Frontend Engineer with 3 years experience in React, TypeScript, and CSS3."
            ],
            "model_version": "v5"
        }
        res = self.client.post('/score', 
                               data=json.dumps(payload), 
                               content_type='application/json',
                               headers={'HX-Request': 'true'})
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn('candidate-row', html)
        self.assertIn('Matched Skills', html)
        self.assertIn('Ensemble Score', html)

    def test_validation_errors(self):
        """Test input validation for short job desc or missing resumes."""
        res = self.client.post('/score', data=json.dumps({"job_desc": "Short", "resumes": ["test"]}), content_type='application/json')
        self.assertEqual(res.status_code, 400)

        res2 = self.client.post('/score', data=json.dumps({"job_desc": "Long enough job description here but no resumes", "resumes": []}), content_type='application/json')
        self.assertEqual(res2.status_code, 400)

    def test_feedback_and_csv_export(self):
        """Test POST /feedback and GET /export-csv."""
        res_fb = self.client.post('/feedback', 
                                  data=json.dumps({"candidate_id": "Candidate 1", "feedback": "approve"}),
                                  content_type='application/json')
        self.assertEqual(res_fb.status_code, 200)

        res_csv = self.client.get('/export-csv')
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn('text/csv', res_csv.content_type)
        csv_text = res_csv.data.decode('utf-8')
        self.assertIn('Candidate ID', csv_text)

    def test_upload_file_docx(self):
        """Test POST /upload with a real docx document."""
        from docx import Document
        doc = Document()
        doc.add_paragraph("Machine learning engineer with 4 years Python and PyTorch experience.")
        bio = io.BytesIO()
        doc.save(bio)
        bio.seek(0)

        data = {
            'file': (bio, 'test_resume.docx')
        }
        res = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        res_json = json.loads(res.data)
        self.assertEqual(res_json.get('status'), 'success')
        self.assertIn('Machine learning engineer', res_json.get('text'))

    def test_scoring_with_named_dict_candidates(self):
        """Test POST /score with candidate metadata dicts preserving custom names."""
        payload = {
            "job_desc": "Senior Python and AI Engineer with experience in PyTorch, transformers, and NLP.",
            "resumes": [
                {
                    "name": "Alex Chen (Senior AI Engineer)",
                    "text": "Senior AI Engineer with 4 years experience in Python, PyTorch, transformers, and NLP."
                },
                {
                    "name": "Marcus Vance (Chef)",
                    "text": "Executive chef with culinary experience in kitchen management and recipes."
                }
            ],
            "model_version": "v5"
        }
        res = self.client.post('/score', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(len(data.get('results')), 2)

        # Names should be preserved
        c1 = data['results'][0]
        c2 = data['results'][1]
        self.assertEqual(c1['candidate_id'], "Alex Chen (Senior AI Engineer)")
        self.assertEqual(c2['candidate_id'], "Marcus Vance (Chef)")
        self.assertGreater(c1['score'], c2['score'])
        self.assertEqual(c2['score'], 0.0)

    def test_whitespace_resumes_rejected(self):
        """Test POST /score rejects whitespace or empty resume arrays."""
        payload = {
            "job_desc": "Senior Python and AI Engineer with experience in PyTorch and NLP.",
            "resumes": ["", "   "]
        }
        res = self.client.post('/score', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertEqual(data.get('status'), 'error')

    def test_zero_byte_file_upload(self):
        """Test POST /upload with a 0-byte file returns 400."""
        data = {
            'file': (io.BytesIO(b''), 'empty_file.pdf')
        }
        res = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        res_json = json.loads(res.data)
        self.assertEqual(res_json.get('status'), 'error')

    def test_blank_pdf_no_text_extracted(self):
        """Test POST /upload with blank raster-like PDF returns 400."""
        import PyPDF2
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        bio = io.BytesIO()
        writer.write(bio)
        bio.seek(0)

        data = {
            'file': (bio, 'rasterized_blank.pdf')
        }
        res = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        res_json = json.loads(res.data)
        self.assertEqual(res_json.get('status'), 'error')
        self.assertIn('No text extracted', res_json.get('error'))

    def test_error_format_json_vs_htmx(self):
        """Test error responses are JSON for API clients and HTML for HTMX."""
        # JSON API call
        res_json = self.client.post('/score', data=json.dumps({"job_desc": "Short", "resumes": ["test"]}), content_type='application/json')
        self.assertEqual(res_json.status_code, 400)
        data = json.loads(res_json.data)
        self.assertEqual(data.get('status'), 'error')

        # HTMX call
        res_htmx = self.client.post('/score',
                                    data=json.dumps({"job_desc": "Short", "resumes": ["test"]}),
                                    content_type='application/json',
                                    headers={'HX-Request': 'true'})
        self.assertEqual(res_htmx.status_code, 400)
        html = res_htmx.data.decode('utf-8')
        self.assertIn('<div', html)
        self.assertIn('at least 20 characters', html)

    def test_score_chart_in_htmx_partial(self):
        """Test HTMX partial contains scoreChart canvas and results-chart-data JSON."""
        payload = {
            "job_desc": "Cloud DevOps Engineer with Docker, Kubernetes, Terraform, and Linux administration.",
            "resumes": [
                "DevOps Engineer with 4 years experience in Docker, Kubernetes, and Linux."
            ],
            "model_version": "v5"
        }
        res = self.client.post('/score',
                               data=json.dumps(payload),
                               content_type='application/json',
                               headers={'HX-Request': 'true'})
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn('id="scoreChart"', html)
        self.assertIn('id="results-chart-data"', html)

    def test_zero_em_dashes_in_compact_templates(self):
        """Strict pre-flight audit: verify zero em-dashes (— or –) in all compact templates."""
        templates_dir = os.path.join(BASE_DIR, 'templates')
        for root, _, files in os.walk(templates_dir):
            for file in files:
                if file.endswith('.html'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.assertNotIn('—', content, f"Em-dash found in {file}")
                        self.assertNotIn('–', content, f"En-dash found in {file}")

if __name__ == '__main__':
    unittest.main()
