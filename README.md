# Financial Complaints Analyzer

End-to-end pipeline that turns a financial-institution complaint dataset into a topic taxonomy, a topic classifier, and an agentic triage system for an operations team.

Built for the **CocaColaHBC AI Developer Technical Assessment**.

---

## Two notebooks

| Notebook | What it does |
|---|---|
| [`complaint_analysis.ipynb`](complaint_analysis.ipynb) | **Task 1** — EDA → preprocessing → sentence embeddings → BERTopic + manual taxonomy → Logistic Regression and DistilBERT-LoRA classifiers → model comparison. |
| [`complaint_agent.ipynb`](complaint_agent.ipynb) | **Task 2** — ChromaDB vector store → LangGraph agent with regex-first urgency rules + Gemini fallback → demo on held-out complaints → daily-briefing batch function. |

Task 2 consumes artefacts produced by Task 1 (`outputs/df_clf.parquet`, `outputs/embeddings.npy`, the saved classifier, etc.). Run Task 1 first.

---

## Setup

```powershell
# 1. Create + activate the project venv (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Core dependencies
pip install -r requirements.txt

# 3. PyTorch with CUDA 11.8 (LoRA training; CPU works for inference only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env  — get a free key at
#    https://aistudio.google.com/app/apikey
```

Optional — register the venv as a Jupyter kernel for VS Code:

```powershell
python -m ipykernel install --user --name=fca-venv --display-name "Python (FCA venv)"
```

Pick the **Python (FCA venv)** kernel when you open each notebook.

---

## Data

`complaints_flat.csv` (78,313 rows × 18 columns) — CFPB consumer complaint extract, Chase-heavy. ~27 % of rows have a narrative; all modelling runs on that subset (20,998 docs after preprocessing).

---

## Key design choices

- **BERTopic** with `all-MiniLM-L6-v2` embeddings, UMAP→HDBSCAN clustering, bigram/trigram c-TF-IDF for keyword extraction. 25 topics after `min_cluster_size=100`, `min_samples=3`, and post-hoc outlier reduction.
- **Hybrid feature set** for Logistic Regression (TF-IDF + sentence embeddings + product/channel metadata). Outperformed DistilBERT-LoRA by ~11 macro-F1 points on this dataset (0.79 vs 0.71); ships as the production classifier.
- **Agent** is a LangGraph state machine: classify → assess urgency → retrieve similar past complaints → draft case note → review gate. Regex urgency rules fire first (zero LLM tokens for the obvious HIGH cases); Gemini 2.5 Flash Lite handles the MEDIUM/LOW grey zone.
- **HITL routing** triggers on confidence < 0.60, urgency = HIGH, or no similar precedents found.
- **Vector DB** indexes Task 1's train + val splits; the demo set is Task 1's test split (held out from both classifier *and* ChromaDB).

---

## Outputs directory

`outputs/` holds embeddings, models, ChromaDB index, demo complaints, plots. Large rebuildable artefacts (`chroma_db/`, `lora_checkpoints/`) are gitignored — they regenerate when the relevant cells run.
