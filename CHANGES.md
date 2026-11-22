# Changes Made to RankWise

Summary of everything changed after cloning the repo, so you can explain it (and defend it) to your panel.

Commit: `1dd88af` — "Fix benchmark integrity, Indian phone parsing, and install docs"
Pushed to: https://github.com/ayush-bit-spec/RankWise (remote `myfork`)

---

## 1. Setup / environment (not in the commit, local only)

| What | Details |
| :--- | :--- |
| `backend/venv` | Python 3.12 virtual environment (gitignored) |
| `backend/.env` | Copied from `.env.example`; `LLM_PROVIDER=mock`, no API keys (gitignored) |
| Demo data | Ran `python scripts/seed.py` to create `rankwise.db` with 2 jobs and 4 candidates (gitignored) |
| `.claude/launch.json` | Local config so the app can start the API (port 8000) and dashboard (port 8501). Not committed. |

## 2. Install problems found and fixed

**Problem A — `blis` failed to build.** `spacy` pulled in an old `blis` that tried to compile from source on Python 3.12.
`spacy` and `faiss-cpu` turned out to be **unused by the code**, so the minimal install skips them.

**Problem B — version conflicts in `requirements.txt`.** `pydantic==2.9.2` and `httpx==0.27.2` were pinned exactly, but the latest `google-genai` needs newer versions of both.
Fix in [backend/requirements.txt](backend/requirements.txt):

- `pydantic==2.9.2` -> `pydantic>=2.9.2`
- `httpx==0.27.2` -> `httpx>=0.28.1,<1.0.0`

**Problem C — the full install is huge and slow** (PyTorch via `sentence-transformers`).
The code already falls back to a built-in vectorizer and reranker when `sentence-transformers` is missing.
Fix: new file [backend/requirements-core.txt](backend/requirements-core.txt), the minimal set the app really imports. It installs in about 90 seconds. It also pins `pyarrow>=14,<18`, because newer `pyarrow` needs NumPy 2 and crashed the dashboard tables.

## 3. Code fixes

### 3a. Benchmark was rigged (most important)
File: [backend/app/core/evaluation/benchmark.py](backend/app/core/evaluation/benchmark.py)

- **Before:** the "Full System" score read each resume's answer key (`ground_truth_relevance`, `is_keyword_trap`). Keyword traps were hard-coded to score 0.05, and "good" resumes were boosted. The result was circular and could never fail.
- **After:** the Full System runs the **real ranking pipeline** (chunking -> two-stage retrieval -> evidence verdict -> deterministic scoring). Labels are used only afterwards to compute the metrics (NDCG, Precision, MRR).
- Experience and education fit get zero weight in the benchmark, because the raw benchmark text has no parsed experience or education.

### 3b. Misleading benchmark text
Same file.

- Summary now says "outperforms", "ties with" or "underperforms" based on the actual numbers (it used to always say "superior" and "successfully filtering").
- Keyword-trap lines say "no improvement over baseline" when the rank didn't change (it used to claim the trap was "penalized to #3" when the baseline rank was also #3).
- Removed the duplicate `NDCG@3` row that appeared for the 3-resume roles.

### 3c. Phone parsing for Indian numbers
File: [backend/app/core/parser/section_heuristics.py](backend/app/core/parser/section_heuristics.py)

- **Before:** the regex only matched the US layout (3-3-4), so `+91 98765 43210` came back empty.
- **After:** added an Indian 5-5 pattern. US formats and plain 10-digit numbers still match. Checked on six sample strings.

### 3d. Dataset size claim
File: [backend/app/core/evaluation/synthetic_data.py](backend/app/core/evaluation/synthetic_data.py) — docstring changed from "45" to "21" resumes (15 + 3 + 3), which is what the data actually contains.

## 4. Documentation fixes
File: [README.md](README.md)

- **Benchmark table replaced** with measured numbers (see below), plus a **Limitations** paragraph.
- **Setup steps corrected:** the old steps assumed a `venv` existed in the repo. They now create one, copy `.env`, and point to `requirements-core.txt`.

## 5. Benchmark: before vs after (main role, 15 resumes)

| System | README claimed (NDCG@5) | Measured now (NDCG@5) |
| :--- | :---: | :---: |
| Baseline ATS (TF-IDF) | 0.812 | 0.917 |
| Embeddings only | 0.894 | 0.835 |
| RankWise Full System | 0.958 | **1.000** (real pipeline) |
| Keyword-stuffer rank | #10 | #10 of 15 (baseline had it at #5) |

On the two 3-resume roles the Full System only ties or modestly beats the baseline, and the keyword-trap resume stays at #3 (no improvement).

## 6. Verification done

- All **92 tests pass** after the changes.
- Live end-to-end test via the API: upload a DOCX -> parsed -> ranked -> bias report -> chat all responded. The test candidate was deleted afterwards and the ranking re-run, so the demo data is back to the original 4.
- Dashboard pages checked: Overview, Job Positions, Candidate Resumes, Semantic Ranking, Bias & Fairness Audit, Evaluation Benchmarks.

## 7. Known limitations (say these before the panel does)

- **Mock LLM by default.** Strengths, gaps and interview questions are canned text (one explanation calls Sarah Chen "Alex"). A real Gemini or OpenAI key in `backend/.env` fixes this.
- **Small synthetic benchmark.** 21 resumes shows the pipeline behaves sensibly. It does not prove real-world accuracy.
- **Fallback text matching.** Without `sentence-transformers`, similarity uses a built-in vectorizer, so numbers will differ from a full neural setup.
- **Dashboard wording.** The Evaluation page still says RankWise "correctly penalizes false evidence". That holds for the 15-resume role but not the two 3-resume roles.
- **Minor, not fixed:** the chat answer repeats its quoted text twice; `AGENTS.md` mentions a root-level `streamlit_app.py` that doesn't exist.

## 8. Files changed

| File | Status |
| :--- | :--- |
| `README.md` | modified |
| `backend/app/core/evaluation/benchmark.py` | modified |
| `backend/app/core/evaluation/synthetic_data.py` | modified |
| `backend/app/core/parser/section_heuristics.py` | modified |
| `backend/requirements.txt` | modified |
| `backend/requirements-core.txt` | new |
| `CHANGES.md` | new (this file, not yet committed) |
