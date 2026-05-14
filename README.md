

<div align="center">

<br/>

# VeritAI

### Fake News Detection and Fact Verification

*Before you share it, let us check it.*

<br/>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20VeritAI-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://ruturajjadhav35-veritai.hf.space)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

</div>

---

## What Is VeritAI

VeritAI is an AI-powered fact-checking tool that reads news articles and tells you whether the claims inside them hold up against what trusted sources actually report.

It started as an MSc Artificial Intelligence dissertation at Queen Mary University of London, built around a simple question: could AI genuinely help people cut through online misinformation? That question turned into a four-stage pipeline, a trained model, and eventually a fully deployed product that anyone can use for free.

You paste in a news article or a URL. VeritAI reads it, pulls out the key factual claims, checks them against Reuters, the BBC, professional fact-checkers, The Guardian, and Wikipedia, and gives you a straight answer on whether those claims hold up. Every source it checked is linked so you can read the evidence yourself.

---

## Try It Live

The app is live and free to use. No account, no sign-up, no limit.

**[ruturajjadhav35-veritai.hf.space](https://ruturajjadhav35-veritai.hf.space)**

---

## Why This Exists

Misinformation does not spread because people want to be deceived. It spreads because it is fast, convincing, and everywhere, and most people do not have time to fact-check every article they read.

Professional fact-checkers like Reuters, AP, and PolitiFact do invaluable work, but they can only cover a fraction of what circulates online. VeritAI is not trying to replace them. It is trying to make their work, and the broader body of trustworthy reporting, more accessible to anyone who wants to check something before they share it.

---

## How It Works

When you submit an article, VeritAI runs it through four stages in sequence.

The first stage puts the article through a BERT language model that was trained on nearly 45,000 news articles, roughly half of them real reporting from Reuters and half fabricated content from sources flagged as unreliable. BERT learned to tell them apart not by looking things up, but by recognising how they are written. Sensational language, vague sourcing, conspiratorial framing, and manufactured urgency all leave patterns in the text. BERT reads the opening of your article and makes an initial call.

The second stage uses spaCy to find the sentences worth checking. Not every sentence in an article can be fact-checked. Opinions, feelings, and vague claims do not have a right or wrong answer. spaCy identifies the sentences that make specific, testable claims about real people, organisations, places, dates, and events. Those are the sentences worth verifying.

The third stage searches four trusted source networks simultaneously for evidence on each claim. Professional fact-checkers carry the most weight because if someone like PolitiFact or Snopes has already investigated a claim, their verdict is the most reliable evidence available. Major news outlets come next, followed by The Guardian's full article archive and Wikipedia as background context.

The fourth and final stage uses RoBERTa-MNLI to read each claim and its evidence together and decide whether the evidence supports the claim, contradicts it, or is insufficient to say either way. This is not keyword matching. RoBERTa understands the semantic relationship between two pieces of text and whether one logically follows from the other.

---

## The Model

The BERT classifier at the heart of VeritAI was fine-tuned on the ISOT Fake News Dataset, a collection of 44,898 labelled articles split 70 percent for training, 15 percent for validation, and 15 percent for testing. It uses the bert-base-uncased architecture with a custom classification head, trained with the AdamW optimiser at a learning rate of 1e-5, reading the first 128 tokens of each article.

The model achieved 92 percent accuracy on the held-out test set.

Rather than always returning a confident verdict, VeritAI uses a three-tier confidence system. When BERT's confidence is high and the source evidence agrees, it returns a strong verdict. When signals are mixed, it flags the uncertainty honestly and encourages you to check the sources yourself. A tool that pretends to be certain when it is not is less useful than one that tells you when to look closer.

---

## The Sources

VeritAI queries four source networks for every claim, each weighted by how reliable they are for fact-checking purposes.

Google Fact Check carries the highest weight because it aggregates verdicts from professional fact-checking organisations including PolitiFact, Snopes, Reuters Fact Check, AP Fact Check, and FullFact. If a claim has already been formally investigated, that verdict is the strongest evidence available.

NewsAPI is searched next, restricted to trusted outlets only. The query goes to Reuters, BBC, Associated Press, The New York Times, Bloomberg, The Guardian, and CNN. Consistent coverage of the same story across multiple major outlets is a strong signal that a claim reflects what actually happened.

The Guardian's official API provides access to full article text rather than just headlines, giving one independent quality-journalism source in the mix to balance the aggregated results.

Wikipedia provides encyclopedic background context as the baseline reference, particularly useful for established facts, historical events, and information about public figures and organisations.

---

## Running The Project Locally

If you want to run VeritAI on your own machine, you will need Python 3.12 and Git with Git LFS installed.

Clone the repository and set up a virtual environment, then install PyTorch with the CPU build before installing the rest of the dependencies. The full instructions are below.

```bash
git clone https://github.com/Ruturajjadhav35/Context-Aware-Fake-News-Detection-and-Fact-Verification-
cd Context-Aware-Fake-News-Detection-and-Fact-Verification-

python3.12 -m venv venv
source venv/bin/activate

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env
uvicorn main:app --reload --port 8000
```

Open your browser at `http://localhost:8000`.

The app requires three API keys to query the evidence sources. These go in a `.env` file that is never committed to the repository. A template called `.env.example` is included showing the exact variable names needed. All three keys are available on free tiers from their respective providers.

Do not share your `.env` file or commit it to any public repository.

---

## Honest Limitations

VeritAI is a research and assistance tool, not a truth oracle. These limitations are worth knowing before relying on any verdict it gives.

The training data is from 2016 and 2017. The ISOT dataset reflects the misinformation landscape of that era, which is broadly similar to today but not identical. Some newer tactics may not be recognised as reliably.

The pipeline is English only and has not been tested on other languages.

Breaking news is difficult. If something just happened, no source may have covered it yet, which limits how much the verification stage can find.

Satire and opinion pieces are not designed to be factually accurate. Submitting them will often produce unreliable or misleading results.

Very short article snippets may not contain enough named entities for the claim extraction stage to find anything worth checking.

None of this means the tool is not useful. It means you should treat its verdict as a strong signal and follow the source links to make your own judgment.

---

## What Is Coming Next

The core pipeline is built and deployed. The next phase is making it smarter over time.

Every article analysed by VeritAI is stored in a database. High-confidence verdicts from professional fact-checkers will be used as training signal to fine-tune the BERT model on modern, real-world content. The goal is a system that improves continuously as more people use it.

Beyond that, integrating the LIAR dataset will broaden the model's coverage beyond political news, and a browser extension would let people check articles without leaving the page they are reading.

---

## About The Author

My name is Ruturaj Jadhav. I built VeritAI as part of my MSc in Artificial Intelligence at Queen Mary University of London. The dissertation asked whether a multi-stage AI pipeline could provide meaningful, transparent fact-checking rather than just a black-box label. I think the answer is yes, and there is a lot more still to build.

If you have questions about the project, want to discuss the methodology, or are interested in collaborating, I would genuinely like to hear from you.

[![GitHub](https://img.shields.io/badge/GitHub-Ruturajjadhav35-181717?style=flat-square&logo=github)](https://github.com/Ruturajjadhav35)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ruturaj%20Jadhav-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/ruturaj-jadhav11)
[![Email](https://img.shields.io/badge/Email-Get%20In%20Touch-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:ruturajjadhav5338@gmail.com)

---

<div align="center">

Built with PyTorch, HuggingFace Transformers, FastAPI, spaCy, and Docker

MSc Artificial Intelligence Dissertation, Queen Mary University of London

</div>
