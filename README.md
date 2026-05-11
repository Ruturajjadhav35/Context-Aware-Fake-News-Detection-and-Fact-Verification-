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

# VeritAI — Fake News Detection & Fact Verification

> An AI-powered pipeline that detects misinformation and verifies factual claims in news articles, cross-referencing Wikipedia, Google Fact Check, NewsAPI, and The Guardian.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![BERT](https://img.shields.io/badge/BERT-fine--tuned-orange)](https://huggingface.co/bert-base-uncased)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## How It Works

VeritAI runs a **4-stage pipeline** on every article submitted:

| Stage | Model | What it does |
|-------|-------|-------------|
| ① Classification | BERT (fine-tuned) | Reads first 128 tokens, outputs fake/real verdict |
| ② Claim extraction | spaCy NER | Scores sentences by fact-checkability, picks top 5 |
| ③ Evidence retrieval | 4 APIs in parallel | Wikipedia · Google Fact Check · NewsAPI · Guardian |
| ④ NLI verification | RoBERTa-MNLI | SUPPORTS / REFUTES / NEI verdict per claim |

---

## Model Details

- **Base model**: `bert-base-uncased` (12 layers, 768 hidden dims)
- **Classifier head**: `Linear(768→512) → ReLU → Dropout(0.1) → Linear(512→2) → LogSoftmax`
- **Training data**: ISOT Fake News Dataset — 44,898 articles (70/15/15 split)
- **Sequence length**: 128 tokens
- **Optimiser**: AdamW, lr=1e-5
- **Accuracy**: 92% on held-out test set

---

## Evidence Sources & Weights

| Source | Weight | Coverage |
|--------|--------|----------|
| Google Fact Check | 1.5× | PolitiFact, Snopes, Reuters FC, AP, FullFact |
| NewsAPI | 1.3× | Reuters, BBC, AP, NYT, Bloomberg, Guardian, CNN |
| The Guardian | 1.2× | Full Guardian article text |
| Wikipedia | 1.0× | Encyclopedic background context |

Final verdict per claim is determined by **weighted majority voting** across all sources that returned evidence.

---

## Confidence Tiers

| Tier | Condition | UI |
|------|-----------|----|
| ✅ High confidence | BERT ≥ 85% + source agreement ≥ 60% | Green banner |
| ⚠ Moderate confidence | BERT ≥ 65% + source agreement ≥ 40% | Amber banner |
| ❓ Uncertain | BERT < 65% | Grey "Uncertain" banner |

---

## API Endpoints

```
POST /analyse          — Main inference endpoint
GET  /health           — Model + DB + API status
GET  /queries          — List stored queries (filterable by verdict)
GET  /queries/{id}     — Single query with full claim data
GET  /stats            — Summary stats
```

### Example request

```bash
curl -X POST https://your-space.hf.space/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste your article text here..."}'
```

### Example response

```json
{
  "verdict": "fake",
  "confidence": 87.4,
  "total_claims": 5,
  "conflict_pct": 60,
  "claims": [
    {
      "text": "Scientists confirmed 5G towers transmit mind-control frequencies",
      "verdict": "refutes",
      "conf": 91,
      "evidence": "No scientific evidence supports this claim..."
    }
  ]
}
```

---

## Project Structure

```
VeritAI/
├── main.py                  # FastAPI app — all endpoints + pipeline
├── static/
│   └── index.html           # Frontend — responsive, dark mode
├── model/
│   ├── bert_fake_news.pt    # Fine-tuned BERT weights
│   └── tokenizer/           # Saved BERT tokenizer
├── Dockerfile               # HuggingFace Spaces deployment
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
└── veritai.db               # SQLite database (persisted at /data/)
```

---

## Environment Variables

Create a `.env` file (locally) or set Secrets in HuggingFace Spaces settings:

```env
NEWS_API_KEY=your_newsapi_key
GOOGLE_FACTCHECK_API_KEY=your_google_key
GUARDIAN_API_KEY=your_guardian_key
DB_PATH=/data/veritai.db
```

> **Never commit your `.env` file.** Add it to `.gitignore`.

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/Ruturajjadhav35/VeritAI
cd VeritAI

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install CPU-only PyTorch first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Run the server
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Deploying to HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repo to the Space's Git remote
4. Add your API keys under **Settings → Repository secrets**:
   - `NEWS_API_KEY`
   - `GOOGLE_FACTCHECK_API_KEY`
   - `GUARDIAN_API_KEY`
5. The Space will build automatically — first build takes ~5 minutes

> **Model file size**: `bert_fake_news.pt` is ~440MB. Use [Git LFS](https://git-lfs.com) to push it:
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> git add model/bert_fake_news.pt
> git commit -m "Add model weights via LFS"
> git push
> ```

---

## Limitations

- BERT classification is influenced by writing style, not just factual content
- NLI models can struggle with implicit negation and complex reasoning
- Real-time breaking news may have no evidence in any source yet
- Performs best on English-language political and general news
- Satire, opinion pieces, and highly technical content may give unreliable results

---

## Author

**Ruturaj Jadhav**  
MSc Artificial Intelligence — Queen Mary University of London  
[GitHub](https://github.com/Ruturajjadhav35) · [LinkedIn](https://linkedin.com/in/ruturaj-jadhav11) · [Email](mailto:ruturajjadhav5338@gmail.com)

---

## License

MIT — see [LICENSE](LICENSE) for details.
