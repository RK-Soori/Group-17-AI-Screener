# Group 17 - AI Candidate Screening System (v5.0 Compact Edition)

> **Design Read**: High-throughput AI Talent Screening & Technical Recommendation Cockpit for engineering leaders, with precision telemetry and high-signal data density, leaning toward Linear-inspired dark-slate tactile engineering system.

---

## 1. 🎛️ The Three Dials Configuration
- **`DESIGN_VARIANCE`**: `7/10` (Asymmetric 50/50 split-screen hero, dynamic section cadence, anti-center layout)
- **`MOTION_INTENSITY`**: `5/10` (Tactile spring physics, micro-interactions `active:scale-[0.98]`, reduced-motion fallback)
- **`VISUAL_DENSITY`**: `7/10` (High-density telemetry dashboard, compact metrics, zero wasted vertical whitespace)

---

## 2. 🎨 Design Tokens & Anti-Slop Discipline
- **Accent Color**: Single accent `#10b981` (Precision Emerald, saturation ~76%), zero AI-purple glow, zero warm cream/clay.
- **Base Theme**: Globally locked dark cockpit (`#090d16` background, `#0f172a` surfaces, `#1e293b` borders).
- **Typography**: Sans display `Geist` / `Outfit`, Monospace `Geist Mono` / `JetBrains Mono`.
- **Corner Radius**: Uniform lock to `rounded-xl` (`12px` / `0.75rem`) across cards, buttons, badges, inputs, and drawers.
- **Accessibility & Contrast**: Strict WCAG AA compliance (>= 4.5:1) verified on all text and inputs.
- **Zero Em-Dashes**: All dashes rendered as standard hyphens `-` across headlines, labels, body, and buttons.

---

## 3. 📐 Viewport & Layout Architecture
- **Hero Viewport**: `min-h-[100dvh]` with top padding capped at `pt-20`.
- **Hero Content Budget (Strictly 4 Elements)**:
  1. Eyebrow label: `AI Screening Engine v5.0`
  2. Headline: `Automated Candidate Screening & Technical Ranking` (2 lines)
  3. Subtext: `Hybrid SBERT and TF-IDF matching engine calibrated for unbiased engineering talent acquisition.` (13 words)
  4. CTAs: `Run Screening` (primary) + `View Telemetry` (secondary)
- **Anti-Center Asymmetric 50/50 Layout**: Left column features hero budget elements; right column features real interactive matching engine telemetry and weight simulator.
- **Eyebrow Ratio Restraint**: Exactly 1 eyebrow label across 4 total sections (<= `ceil(4/3) = 2`).
- **Desktop Navigation**: Single-line rendering at `lg` with max height 64px (`h-16`).

---

## 4. ⚡ Features & Endpoints
- **1-Click Preset Requisitions**: Instantly populate job requisitions and candidate dossiers for AI Engineer, React Frontend Lead, and Cloud DevOps.
- **Multi-Format Ingestion**: Supports Drag & Drop PDF and DOCX file parsing via `PyPDF2` and `python-docx`.
- **Hybrid Matching Ensemble (V5.0)**:
  - Fine-Tuned Domain SBERT (65% weight) with 140-word overlapping sliding chunks
  - Domain-Adapted TF-IDF (20% weight, 48,393 features)
  - Bounded Skill Taxonomy (15% weight, 143+ competencies across 4 clusters)
  - Zero-Match Technical Gatekeeper
- **Real-Time Client Filters**: Instant search by candidate ID or skill keyword, plus decision tier filtering pills (`Highly Suitable`, `Suitable`, `Low Match`).
- **Active Learning Loop**: Recruiter thumbs up/down feedback logging to `feedback.json`.
- **One-Click CSV Export**: Download complete candidate evaluation metrics to CSV.

---

## 5. 🚀 How to Run Both Versions Side-by-Side

### Run v4.1 (Classic Interface - Completely Intact)
```bash
cd "c:\Users\Kavinda\Desktop\sem 4\AI\group project ai\web_app\releases\v4.1"
python app.py
# Running on http://127.0.0.1:5000
```

### Run v5.0 Compact (Compact Cockpit Edition)
```bash
cd "c:\Users\Kavinda\Desktop\sem 4\AI\group project ai\web_app\releases\v5.0_compact"
python app.py --port 5001
# Running on http://127.0.0.1:5001
```

Both releases can run concurrently on ports 5000 and 5001 without port conflicts or file contention.

---

## 6. 🧪 Running the Verification Test Suite
```bash
cd "c:\Users\Kavinda\Desktop\sem 4\AI\group project ai\web_app\releases\v5.0_compact"
python -m unittest test_v5_compact.py
```
