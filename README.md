---
title: VeritAI
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
license: mit
app_port: 7860
short_description: AI-powered fake news detection and fact verification
tags:
  - nlp
  - bert
  - fake-news
  - fact-checking
  - misinformation
  - transformers
  - spacy
  - roberta
---

<div align="center">

# VeritAI
### Fake News Detection & Fact Verification

*Before you share it — let us check it.*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow?style=for-the-badge&logo=huggingface)](https://ruturajjadhav35-veritai.hf.space)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)](LICENSE)

</div>

---

## What is VeritAI?

VeritAI is an AI-powered fact-checking tool that analyses news articles and tells you whether they hold up against what trusted sources actually report.

Paste in any article — or drop a URL — and within seconds VeritAI will:

- **Classify** the writing style as consistent with real or fabricated content
- **Extract** the key factual claims worth checking
- **Search** Reuters, BBC, Wikipedia, Google Fact Check, and The Guardian
- **Verify** each claim against the evidence and give you a plain-English verdict

Built as an MSc Artificial Intelligence dissertation at **Queen Mary University of London**.

---

## Live Demo

🔗 **[Try VeritAI live →](https://ruturajjadhav35-veritai.hf.space)**

---

## How It Works

VeritAI runs a four-stage pipeline on every article submitted:

```
Article input
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1 — BERT Classification  │
│  Reads writing style & framing  │
│  Output: Fake / Real + confidence│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 2 — spaCy Claim Extract  │
│  Finds specific, testable facts │
│  Output: Top 5 checkable claims │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 3 — Evidence Retrieval   │
│  Searches 4 source networks     │
│  in parallel                    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 4 — RoBERTa-MNLI         │
│  Reads claim + evidence together│
│  Output: Supports/Refutes/NEI   │
└─────────────────────────────────┘
```

### Evidence Sources & Weights

| Source | Weight | What it covers |
|--------|--------|----------------|
| 🔍 Google Fact Check | 1.5× | PolitiFact, Snopes, Reuters FC, AP, FullFact |
| 📰 NewsAPI | 1.3× | Reuters, BBC, AP, NYT, Bloomberg, Guardian, CNN |
| 🛡️ The Guardian | 1.2× | Full Guardian article text via official API |
| 📚 Wikipedia | 1.0× | Encyclopedic background and context |

Final verdicts use **weighted majority voting** across all sources that returned evidence.

---

## Model Details

| Component | Details |
|-----------|---------|
| **Base model** | `bert-base-uncased` |
| **Architecture** | 12 transformer layers, 768 hidden dims |
| **Classifier head** | Linear(768→512) → ReLU → Dropout(0.1) → Linear(512→2) |
| **Training data** | ISOT Fake News Dataset — 44,898 articles |
| **Train/Val/Test split** | 70 / 15 / 15 |
| **Sequence length** | 128 tokens |
| **Optimiser** | AdamW, lr=1e-5 |
| **Test accuracy** | **92%** |
| **NLI model** | `roberta-large-mnli` |
| **Claim extractor** | `spaCy en_core_web_sm` |

---

## Confidence Tiers

Rather than always giving a binary verdict, VeritAI is honest about uncertainty:

| Tier | Condition | What it means |
|------|-----------|---------------|
| ✅ Strong signal | BERT ≥ 85% + source agreement ≥ 60% | High confidence verdict |
| ⚠️ Moderate signal | BERT ≥ 65% + source agreement ≥ 40% | Treat with some caution |
| ❓ Uncertain | BERT < 65% | Evidence is mixed — check sources yourself |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML framework** | PyTorch 2.2.2 |
| **NLP models** | HuggingFace Transformers, spaCy |
| **Backend** | FastAPI + Uvicorn |
| **Database** | SQLite (persistent on HuggingFace Spaces) |
| **Frontend** | Vanilla HTML/CSS/JS — fully responsive |
| **Deployment** | Docker on HuggingFace Spaces (CPU free tier) |

---

## API Endpoints

The backend exposes a REST API you can query directly:

```bash
# Analyse an article
curl -X POST https://ruturajjadhav35-veritai.hf.space/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste your article text here..."}'

# Health check
curl https://ruturajjadhav35-veritai.hf.space/health

# Usage statistics
curl https://ruturajjadhav35-veritai.hf.space/stats

# Query history
curl https://ruturajjadhav35-veritai.hf.space/queries
```

### Example response

```json
{
  "verdict": "fake",
  "confidence": 91.4,
  "total_claims": 5,
  "conflict_pct": 80,
  "claims": [
    {
      "text": "5G towers transmit mind-control frequencies",
      "verdict": "refutes",
      "conf": 94,
      "evidence": "[Reuters: No scientific evidence supports this claim...]"
    }
  ],
  "processing_ms": 4823
}
```

---

## Project Structure

```
VeritAI/
├── main.py                  # FastAPI backend — pipeline + all endpoints
├── static/
│   └── index.html           # Frontend — responsive, dark mode support
├── model/
│   ├── bert_fake_news.pt    # Fine-tuned BERT weights (via Git LFS)
│   └── tokenizer/           # Saved BERT tokenizer
├── Dockerfile               # HuggingFace Spaces deployment
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── .gitignore
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Ruturajjadhav35/Context-Aware-Fake-News-Detection-and-Fact-Verification-
cd Context-Aware-Fake-News-Detection-and-Fact-Verification-

# 2. Create a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install PyTorch (CPU)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 4. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 6. Run the server
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

### Environment Variables

```env
NEWS_API_KEY=your_newsapi_key
GOOGLE_FACTCHECK_API_KEY=your_google_key
GUARDIAN_API_KEY=your_guardian_key
DB_PATH=./veritai.db
```

Get your free API keys from:
- [NewsAPI](https://newsapi.org/register)
- [Google Fact Check Tools](https://console.cloud.google.com)
- [The Guardian Open Platform](https://open-platform.theguardian.com/access)

---

## Limitations

VeritAI is a research tool, not a truth oracle. A few honest caveats:

- **Training data is from 2016–17** — misinformation tactics have evolved since then
- **English only** — the pipeline is not trained for other languages
- **Breaking news** — real-time events may not yet have evidence in any source
- **Satire and opinion** — these aren't meant to be factual and may give unreliable results
- **Writing style bias** — BERT picks up on how articles are written, not just what they say

---

## Roadmap

- [x] BERT classifier — 92% accuracy on ISOT dataset
- [x] Multi-source evidence retrieval (4 APIs)
- [x] RoBERTa-MNLI claim verification
- [x] Responsive web frontend with dark mode
- [x] Deployed on HuggingFace Spaces
- [ ] Continual learning pipeline — retrain on live query data
- [ ] LIAR dataset integration for broader coverage
- [ ] PostgreSQL migration for production scale
- [ ] Browser extension

---

## Author

**Ruturaj Jadhav**
MSc Artificial Intelligence — Queen Mary University of London

[![GitHub](https://img.shields.io/badge/GitHub-Ruturajjadhav35-black?style=flat&logo=github)](https://github.com/Ruturajjadhav35)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ruturaj--jadhav11-blue?style=flat&logo=linkedin)](https://linkedin.com/in/ruturaj-jadhav11)
[![Email](https://img.shields.io/badge/Email-ruturajjadhav5338@gmail.com-red?style=flat&logo=gmail)](mailto:ruturajjadhav5338@gmail.com)

---

<div align="center">
<sub>Built with PyTorch · Transformers · FastAPI · spaCy · Docker</sub>
</div>
