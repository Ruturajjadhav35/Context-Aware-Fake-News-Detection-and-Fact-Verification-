from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
import wikipedia
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
import traceback
import time
import json
import os
import sqlite3
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# ── API keys from environment ──────────────────────────────────────────────────
NEWS_API_KEY             = os.getenv("NEWS_API_KEY", "")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
GUARDIAN_API_KEY         = os.getenv("GUARDIAN_API_KEY", "")

# ── Model & DB paths from environment (overridden on HuggingFace Spaces) ───────
DB_PATH        = os.getenv("DB_PATH",        "./veritai.db")
MODEL_PATH     = os.getenv("MODEL_PATH",     "./model/bert_fake_news.pt")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./model/tokenizer")

# ── Absolute base directory — safe inside Docker ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="VeritAI – Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SQLite database ────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS article_queries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    DEFAULT (datetime('now')),
                article_text    TEXT    NOT NULL,
                article_length  INTEGER,
                verdict         TEXT,
                confidence      REAL,
                total_claims    INTEGER,
                conflict_pct    INTEGER,
                claims          TEXT,
                sources_checked TEXT,
                processing_ms   INTEGER
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_verdict    ON article_queries(verdict);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON article_queries(created_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON article_queries(confidence);")
        conn.commit()
        cur.close()
        conn.close()
        print(f"✓ SQLite database ready at {DB_PATH}")
        return True
    except Exception as e:
        print(f"✗ Database init failed: {e}")
        return False

def save_query(article_text, verdict, confidence, total_claims, conflict_pct,
               claims, sources_checked, processing_ms):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO article_queries (
                article_text, article_length, verdict, confidence,
                total_claims, conflict_pct, claims, sources_checked, processing_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article_text, len(article_text), verdict, confidence,
            total_claims, conflict_pct,
            json.dumps(claims), json.dumps(sources_checked), processing_ms
        ))
        new_id = cur.lastrowid
        conn.commit()
        cur.close()
        conn.close()
        print(f"✓ Query saved — id={new_id}")
        return new_id
    except Exception as e:
        print(f"✗ Failed to save query: {e}")
        return None

# ── BERT model architecture ────────────────────────────────────────────────────
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(0.1)
        self.relu    = nn.ReLU()
        self.fc1     = nn.Linear(768, 512)
        self.fc2     = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

# ── Load all models at startup ─────────────────────────────────────────────────
print("Loading models — this may take a minute...")

try:
    bert_base  = AutoModel.from_pretrained('bert-base-uncased')
    bert_model = BERT_Arch(bert_base)
    bert_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    bert_model.eval()
    bert_tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    print("✓ BERT classifier loaded")
except Exception as e:
    print(f"✗ BERT load failed: {e}")
    bert_model = bert_tokenizer = None

try:
    nli_model     = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
    nli_tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
    nli_model.eval()
    print("✓ RoBERTa-MNLI loaded")
except Exception as e:
    print(f"✗ RoBERTa load failed: {e}")
    nli_model = nli_tokenizer = None

try:
    nlp = spacy.load('en_core_web_sm')
    print("✓ spaCy loaded")
except Exception as e:
    print(f"✗ spaCy load failed: {e}")
    nlp = None

DB_AVAILABLE = init_db()

print(f"  NewsAPI key:          {'✓' if NEWS_API_KEY else '✗ missing'}")
print(f"  Google FactCheck key: {'✓' if GOOGLE_FACTCHECK_API_KEY else '✗ missing'}")
print(f"  Guardian key:         {'✓' if GUARDIAN_API_KEY else '✗ missing'}")
print("All models loaded.\n")

# ── Request model ──────────────────────────────────────────────────────────────
class ArticleRequest(BaseModel):
    text: str

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ── BERT classification ────────────────────────────────────────────────────────
def classify_article(text):
    if bert_model is None or bert_tokenizer is None:
        return 'real', 50.0
    tokens = bert_tokenizer(
        [text], max_length=128, padding=True,
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        output = bert_model(tokens['input_ids'], tokens['attention_mask'])
    probs     = torch.exp(output)
    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()
    print(f"BERT → Real: {real_prob:.4f}  Fake: {fake_prob:.4f}")
    if fake_prob > 0.5:
        return 'fake', round(fake_prob * 100, 1)
    return 'real', round(real_prob * 100, 1)

# ── spaCy claim extraction ─────────────────────────────────────────────────────
def extract_claims(text):
    if nlp is None:
        sentences = [s.strip() for s in text.split('.') if len(s.split()) > 6]
        return sentences[:5]
    doc    = nlp(text)
    scored = []
    for sent in doc.sents:
        words = sent.text.split()
        if len(words) < 6 or len(words) > 60:
            continue
        score    = 0
        entities = [ent.label_ for ent in sent.ents]
        if 'PERSON' in entities: score += 3
        if 'ORG'    in entities: score += 2
        if 'GPE'    in entities: score += 2
        if 'DATE'   in entities: score += 1
        if 'EVENT'  in entities: score += 2
        if 'NORP'   in entities: score += 1
        if 'MONEY'  in entities: score += 1
        if 'LAW'    in entities: score += 2
        if any(w in sent.text.lower() for w in
               ['believe', 'think', 'feel', 'seem', 'appear', 'suggest', 'allegedly']):
            score -= 1
        has_verb = any(tok.pos_ == 'VERB' for tok in sent)
        if not has_verb:
            continue
        if score > 0:
            scored.append((score, sent.text.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:5]]

# ── Evidence sources ───────────────────────────────────────────────────────────
def fetch_wikipedia(claim):
    try:
        results = wikipedia.search(claim, results=3)
        if not results:
            return None, "Wikipedia"
        page = wikipedia.page(results[0], auto_suggest=False)
        return page.summary[:600], f"Wikipedia: {page.title}"
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.summary[:600], f"Wikipedia: {page.title}"
        except Exception:
            return None, "Wikipedia"
    except Exception:
        return None, "Wikipedia"

def fetch_newsapi(claim):
    if not NEWS_API_KEY:
        return None, "NewsAPI"
    try:
        trusted_sources = (
            "reuters,bbc-news,associated-press,the-guardian-uk,"
            "the-wall-street-journal,bloomberg,abc-news,cbs-news,cnn,nbc-news"
        )
        params = {
            "q":        claim[:200],
            "sources":  trusted_sources,
            "language": "en",
            "sortBy":   "relevancy",
            "pageSize": 3,
            "apiKey":   NEWS_API_KEY,
        }
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params=params, headers=HEADERS, timeout=8
        )
        if response.status_code != 200:
            return None, "NewsAPI"
        data     = response.json()
        articles = data.get("articles", [])
        if not articles:
            return None, "NewsAPI"
        snippets = []
        sources  = set()
        for a in articles[:3]:
            source = a.get("source", {}).get("name", "Unknown")
            title  = a.get("title", "")
            desc   = a.get("description", "") or ""
            snippets.append(f"{source}: {title}. {desc}")
            sources.add(source)
        evidence    = ' | '.join(snippets)[:700]
        source_list = ', '.join(list(sources)[:5])
        return evidence, f"NewsAPI ({source_list})"
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return None, "NewsAPI"

def fetch_google_factcheck(claim):
    if not GOOGLE_FACTCHECK_API_KEY:
        return None, "Google FactCheck"
    try:
        params = {
            "query":        claim[:200],
            "key":          GOOGLE_FACTCHECK_API_KEY,
            "languageCode": "en",
            "pageSize":     5,
        }
        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params=params, timeout=8
        )
        if response.status_code != 200:
            return None, "Google FactCheck"
        data   = response.json()
        claims = data.get("claims", [])
        if not claims:
            return None, "Google FactCheck"
        snippets   = []
        publishers = set()
        for c in claims[:3]:
            text    = c.get("text", "")
            reviews = c.get("claimReview", [])
            if reviews:
                review    = reviews[0]
                publisher = review.get("publisher", {}).get("name", "Unknown")
                rating    = review.get("textualRating", "")
                title     = review.get("title", "")
                snippets.append(
                    f"[{publisher} rated this '{rating}'] {title}. Original claim: {text}"
                )
                publishers.add(publisher)
        if not snippets:
            return None, "Google FactCheck"
        evidence       = ' | '.join(snippets)[:700]
        publisher_list = ', '.join(list(publishers)[:4])
        return evidence, f"Fact-checkers ({publisher_list})"
    except Exception as e:
        print(f"Google FactCheck error: {e}")
        return None, "Google FactCheck"

def fetch_guardian(claim):
    if not GUARDIAN_API_KEY:
        return None, "Guardian"
    try:
        params = {
            "q":           claim[:200],
            "show-fields": "trailText,bodyText",
            "page-size":   3,
            "api-key":     GUARDIAN_API_KEY,
        }
        response = requests.get(
            "https://content.guardianapis.com/search",
            params=params, timeout=8
        )
        if response.status_code != 200:
            return None, "Guardian"
        data    = response.json()
        results = data.get("response", {}).get("results", [])
        if not results:
            return None, "Guardian"
        snippets = []
        for r in results[:2]:
            title  = r.get("webTitle", "")
            fields = r.get("fields", {})
            trail  = fields.get("trailText", "")
            body   = fields.get("bodyText", "")[:300]
            snippets.append(f"Guardian: {title}. {trail} {body}")
        evidence = ' | '.join(snippets)[:700]
        return evidence, "The Guardian"
    except Exception as e:
        print(f"Guardian error: {e}")
        return None, "Guardian"

def fetch_multi_source_evidence(claim):
    all_evidence = []
    sources_to_try = [
        (fetch_google_factcheck, 1.5),
        (fetch_newsapi,          1.3),
        (fetch_guardian,         1.2),
        (fetch_wikipedia,        1.0),
    ]
    for source_fn, weight in sources_to_try:
        try:
            ev, src = source_fn(claim)
            if ev:
                all_evidence.append({"source": src, "text": ev, "weight": weight})
        except Exception as e:
            print(f"Source {source_fn.__name__} failed: {e}")
            continue
    return all_evidence

# ── NLI verification ───────────────────────────────────────────────────────────
def run_nli(claim, evidence):
    if nli_model is None or nli_tokenizer is None:
        return 'nei', 50.0
    try:
        tokens = nli_tokenizer(
            claim, evidence,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            output = nli_model(**tokens)
        probs  = torch.softmax(output.logits, dim=1)[0]
        labels = ['refutes', 'nei', 'supports']
        idx    = torch.argmax(probs).item()
        return labels[idx], round(torch.max(probs).item() * 100, 1)
    except Exception as e:
        print(f"NLI error: {e}")
        return 'nei', 50.0

def verify_claim(claim):
    evidence_items = fetch_multi_source_evidence(claim)
    if not evidence_items:
        return 'nei', 50.0, 'No evidence found across checked sources.', []
    nli_results = []
    for item in evidence_items:
        verdict, conf = run_nli(claim, item['text'])
        nli_results.append({
            "source":   item['source'],
            "verdict":  verdict,
            "conf":     conf,
            "weighted": conf * item['weight'],
            "snippet":  item['text'][:200] + '...' if len(item['text']) > 200 else item['text']
        })
    vote_scores = {'supports': 0.0, 'refutes': 0.0, 'nei': 0.0}
    for r in nli_results:
        vote_scores[r['verdict']] += r['weighted']
    final_verdict = max(vote_scores, key=vote_scores.get)
    total_weight  = sum(vote_scores.values())
    final_conf    = (
        round((vote_scores[final_verdict] / total_weight) * 100, 1)
        if total_weight > 0 else 50.0
    )
    best_evidence = max(nli_results, key=lambda x: x['weighted'])
    sources_str   = ' | '.join(set(r['source'] for r in nli_results))
    evidence_text = f"[Sources: {sources_str}] {best_evidence['snippet']}"
    return final_verdict, final_conf, evidence_text, nli_results

# ── /analyse ──────────────────────────────────────────────────────────────────
@app.post('/analyse')
async def analyse(req: ArticleRequest):
    text       = req.text.strip()
    start_time = time.time()
    if not text:
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Article text is too short to analyse.")
    try:
        verdict, confidence = classify_article(text)
        claims_text         = extract_claims(text)
        claims = []
        for claim in claims_text:
            v, conf, evidence, source_breakdown = verify_claim(claim)
            claims.append({
                'text':             claim,
                'verdict':          v,
                'conf':             conf,
                'evidence':         evidence,
                'source_breakdown': source_breakdown
            })
        refuted         = sum(1 for c in claims if c['verdict'] == 'refutes')
        conflict_pct    = round(refuted / len(claims) * 100) if claims else 0
        processing_ms   = int((time.time() - start_time) * 1000)
        sources_checked = [
            'Google Fact Check',
            'NewsAPI (Reuters/BBC/AP/NYT)',
            'The Guardian',
            'Wikipedia'
        ]
        saved_id = save_query(
            article_text    = text,
            verdict         = verdict,
            confidence      = confidence,
            total_claims    = len(claims),
            conflict_pct    = conflict_pct,
            claims          = claims,
            sources_checked = sources_checked,
            processing_ms   = processing_ms
        )
        return {
            'verdict':         verdict,
            'confidence':      confidence,
            'claims':          claims,
            'total_claims':    len(claims),
            'conflict_pct':    conflict_pct,
            'sources_checked': sources_checked,
            'processing_ms':   processing_ms,
            'saved_id':        saved_id
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ── /queries ──────────────────────────────────────────────────────────────────
@app.get('/queries')
def get_queries(limit: int = 50, verdict: str = None):
    try:
        conn = get_db()
        cur  = conn.cursor()
        if verdict:
            cur.execute("""
                SELECT id, created_at, article_length, verdict, confidence,
                       total_claims, conflict_pct, processing_ms
                FROM   article_queries WHERE verdict = ?
                ORDER  BY created_at DESC LIMIT ?
            """, (verdict, limit))
        else:
            cur.execute("""
                SELECT id, created_at, article_length, verdict, confidence,
                       total_claims, conflict_pct, processing_ms
                FROM   article_queries
                ORDER  BY created_at DESC LIMIT ?
            """, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"total": len(rows), "queries": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/queries/{query_id}')
def get_query(query_id: int):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM article_queries WHERE id = ?", (query_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Query not found.")
        result = dict(row)
        if result.get('claims'):
            result['claims'] = json.loads(result['claims'])
        if result.get('sources_checked'):
            result['sources_checked'] = json.loads(result['sources_checked'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/stats')
def get_stats():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) AS total_queries,
                SUM(CASE WHEN verdict='fake' THEN 1 ELSE 0 END) AS total_fake,
                SUM(CASE WHEN verdict='real' THEN 1 ELSE 0 END) AS total_real,
                ROUND(AVG(confidence),2)    AS avg_confidence,
                ROUND(AVG(processing_ms),0) AS avg_processing_ms,
                ROUND(AVG(total_claims),2)  AS avg_claims,
                ROUND(AVG(conflict_pct),2)  AS avg_conflict_pct,
                MIN(created_at)             AS first_query,
                MAX(created_at)             AS latest_query
            FROM article_queries
        """)
        summary = dict(cur.fetchone())
        cur.execute("""
            SELECT DATE(created_at) AS date, COUNT(*) AS queries,
                   SUM(CASE WHEN verdict='fake' THEN 1 ELSE 0 END) AS fake,
                   SUM(CASE WHEN verdict='real' THEN 1 ELSE 0 END) AS real
            FROM article_queries GROUP BY DATE(created_at)
            ORDER BY date DESC LIMIT 30
        """)
        daily = [dict(r) for r in cur.fetchall()]
        cur.execute("""
            SELECT verdict, ROUND(AVG(confidence),2) AS avg_conf
            FROM article_queries GROUP BY verdict
        """)
        by_verdict = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"summary": summary, "daily": daily, "by_verdict": by_verdict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    return {
        'status':       'ok',
        'bert_loaded':  bert_model is not None,
        'nli_loaded':   nli_model is not None,
        'spacy_loaded': nlp is not None,
        'db_connected': DB_AVAILABLE,
        'apis': {
            'newsapi':   bool(NEWS_API_KEY),
            'factcheck': bool(GOOGLE_FACTCHECK_API_KEY),
            'guardian':  bool(GUARDIAN_API_KEY),
        }
    }

# ── Frontend — absolute paths, safe inside Docker ─────────────────────────────
app.mount(
    '/static',
    StaticFiles(directory=os.path.join(BASE_DIR, 'static')),
    name='static'
)

@app.get('/')
def root():
    return FileResponse(os.path.join(BASE_DIR, 'static/index.html'))
