# Context-Aware-Fake-News-Detection-and-Fact-Verification-
This is Dissertation project for my MSC in AI course 

This project implements a **3-stage NLP pipeline** to detect and verify misinformation using deep learning and natural language processing. It was developed as part of my MSc AI dissertation at **Queen Mary University of London**.

---

##  Abstract
In the digital era, misinformation and fake news pose a significant threat to public discourse, policy, and societal trust.  
This project proposes a **context-aware pipeline** for automated fake news detection and fact verification. The pipeline integrates:

1. **BERT-based classifier** — to distinguish fake vs. real news.  
2. **spaCy-based claim extractor** — to identify factual claims from articles.  
3. **RoBERTa-MNLI verifier** — to validate claims against evidence retrieved from Wikipedia.  

Unlike traditional binary classifiers, this approach provides both classification and **claim-level validation**, making it suitable for real-world journalism support, policy analysis, and automated auditing:contentReference[oaicite:1]{index=1}.

---

##  System Architecture
News Article
└── [Stage 1: BERT Classifier] → Fake?
├─ No → Label as Real
└─ Yes → [Stage 2: spaCy Claim Extraction] → Claims
└─ [Evidence Retrieval: Wikipedia API]
└─ [Stage 3: RoBERTa-MNLI] → Entailment | Contradiction | Neutral



---

##  Installation

### Requirements
- Python 3.9+
- PyTorch
- Hugging Face Transformers
- spaCy
- scikit-learn
- pandas, numpy, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm


Usage
1. Fake News Detection (Stage 1)
Fine-tune BERT on Fake/True datasets:
!python src/train_classifier.py --train_csv data/Fake.csv --val_csv data/True.csv
2. Claim Extraction (Stage 2)
from claim_extraction import extract_claims
claims = extract_claims("George W. Bush Calls Out Trump For Supporting White Supremacy")
print(claims)
3. Claim Verification (Stage 3)
from verify_mnli import verify_claim
verify_claim("George Walker Bush (born July 6, 1946) is an American politician and businessman who was the 43rd president of the United States from 2001 to 2009. A member of the Republican Party and the eldest son of the 41st president, George H. W. Bush, he served as the 46th governor of Texas from 1995 to 2000.")
#Output: contradiction
 ## **License**
This project is for academic and research purposes. Please cite appropriately if reused.
## **Author**
Ruturaj Sujeet Jadhav
ruturajjadhav5338@gmail.com
