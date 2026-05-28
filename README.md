# Financial Complaints Analyzer

End-to-end pipeline that turns a consumer-financial-complaint dataset into a topic taxonomy, a topic classifier, and an agentic triage system for an operations team.

Built for the **CocaColaHBC AI Developer Technical Assessment**.

---

## Notebooks

| Notebook | What it does |
|---|---|
| [`complaint_analysis.ipynb`](complaint_analysis.ipynb) | **Task 1** — EDA, preprocessing, sentence embeddings, BERTopic taxonomy (25 topics), Logistic Regression and DistilBERT-LoRA classifiers, side-by-side comparison. |
| [`complaint_agent.ipynb`](complaint_agent.ipynb) | **Task 2** — ChromaDB vector store, LangGraph agent (classify → urgency → retrieve → draft → review-gate), demo on held-out complaints, daily-briefing batch function. |

Task 2 reads artefacts produced by Task 1 (`outputs/df_clf.parquet`, `outputs/embeddings.npy`, classifier pkls, optional LoRA adapter). Run Task 1 first.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# CUDA build of PyTorch (RTX 2060 / any modern NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy .env.example to .env and add a free Gemini API key
# https://aistudio.google.com/app/apikey
```

In VS Code, register the venv as a kernel and pick it for both notebooks:

```powershell
python -m ipykernel install --user --name=fca-venv --display-name "Python (FCA venv)"
```

---

## Key design choices

- **BERTopic** with `all-MiniLM-L6-v2` embeddings, UMAP → HDBSCAN clustering, bigram/trigram c-TF-IDF for keyword extraction. Post-hoc outlier reduction reassigns the HDBSCAN noise group so every document carries a topic label. 25 topics in the final taxonomy.
- **Hybrid feature set** for Logistic Regression — TF-IDF (full text) + sentence embeddings (full text) + product/channel metadata one-hots. Outperformed DistilBERT-LoRA by ~11 macro-F1 points on this dataset (**0.79 vs 0.71**); shipped as the production classifier.
- **LangGraph agent** is a state machine with regex-first urgency rules (zero LLM tokens on legal / fraud / bankruptcy / FCRA mentions), Gemini 3.1 Flash Lite for the MEDIUM / LOW grey zone, and three independent HITL triggers (confidence < 0.60, urgency = HIGH, no similar precedents found).
- **Vector DB** indexes Task 1's train + val splits; the demo set is Task 1's test split, held out from both the classifier and ChromaDB.

---

## Results

| Model | Macro F1 | Accuracy | HITL rate |
|---|---:|---:|---:|
| Logistic Regression (production) | **0.79** | 0.80 | 40 % |
| DistilBERT + LoRA | 0.71 | 0.72 | 25 % |

LogReg's higher HITL rate is the desired failure mode — when the classifier is unsure, the agent routes to a human.

---

## Outputs directory

`outputs/` holds embeddings, the BERTopic model, classifier pickles, the LoRA adapter, ChromaDB index, and EDA plots. Large rebuildable artefacts (`chroma_db/`, `lora_checkpoints/`) are gitignored — they regenerate when their cells run.
