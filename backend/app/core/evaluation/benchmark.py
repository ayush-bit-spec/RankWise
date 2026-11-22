"""Benchmark evaluation metrics — computes NDCG@k, Precision@k, and MRR for ablation comparison."""

import math
from typing import Any
import numpy as np

from app.core.embeddings import bi_encoder, reranker
from app.core.embeddings.chunker import chunk_resume
from app.core.embeddings.retriever import two_stage_retriever
from app.core.evaluation.baseline_ats import BaselineATS
from app.core.llm import get_llm_client
from app.core.scoring.engine import (
    compute_evidence_location_weight,
    compute_overall_score,
    compute_recency_weight,
    compute_requirement_score,
)
from app.core.scoring.evidence_judge import judge_evidence
from app.schemas.job import Requirement


def compute_dcg(relevances: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
    return dcg


def compute_ndcg(relevances: list[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    actual_dcg = compute_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0.0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 4)


def compute_precision_at_k(relevances: list[int], k: int, relevant_threshold: int = 2) -> float:
    """Compute Precision@k for candidates meeting or exceeding relevance threshold."""
    if k <= 0:
        return 0.0
    top_k = relevances[:k]
    num_relevant = sum(1 for r in top_k if r >= relevant_threshold)
    return round(num_relevant / k, 4)


def compute_mrr(relevances: list[int], relevant_threshold: int = 2) -> float:
    """Compute Mean Reciprocal Rank (reciprocal rank of the first relevant candidate)."""
    for rank, rel in enumerate(relevances, start=1):
        if rel >= relevant_threshold:
            return round(1.0 / rank, 4)
    return 0.0


class EvaluationHarness:
    """Runs ablation evaluation across Baseline ATS, Embeddings-only, and Full System."""

    def __init__(self):
        self.baseline_ats = BaselineATS()
        self.encoder = bi_encoder

    def run_role_evaluation(self, role_data: dict[str, Any]) -> dict[str, Any]:
        """Run 3-way ablation benchmark on a specific role dataset."""
        candidates = role_data["candidates"]
        jd_text = role_data["job_description"]
        reqs = role_data.get("requirements", [])

        # 1. System A: Baseline ATS (TF-IDF keyword matching)
        ats_results = self.baseline_ats.rank_candidates(jd_text, candidates)
        ats_relevances = [r["ground_truth_relevance"] for r in ats_results]

        # 2. System B: Embeddings Only (Bi-encoder whole document cosine similarity)
        cand_texts = [c["raw_text"] for c in candidates]
        jd_emb = self.encoder.encode([jd_text])
        cand_embs = self.encoder.encode(cand_texts)
        bi_sims = self.encoder.compute_similarity(jd_emb, cand_embs)[0]

        emb_pool = [
            {"cand": candidates[i], "score": float(bi_sims[i])}
            for i in range(len(candidates))
        ]
        emb_pool.sort(key=lambda x: x["score"], reverse=True)
        emb_relevances = [x["cand"]["ground_truth_relevance"] for x in emb_pool]

        # 3. System C: Full System — the REAL pipeline used by RankingService:
        #    chunking -> two-stage retrieval -> LLM evidence verdict -> deterministic scoring.
        #    Ground-truth labels (ground_truth_relevance / is_keyword_trap) are NEVER read
        #    here; they are only used afterwards to compute the metrics.
        full_pool = []
        llm = get_llm_client()
        requirements = [
            Requirement(
                text=r["text"],
                type=r.get("type", "must_have"),
                category=r.get("category", "skill"),
            )
            for r in reqs
        ]
        for c in candidates:
            chunks = chunk_resume({}, c["raw_text"])
            matches = two_stage_retriever.match_requirements_to_chunks(
                chunks=chunks, requirements=requirements, stage1_top_k=5
            )
            must_scores: list[float] = []
            nice_scores: list[float] = []
            for m in matches:
                best_chunk = m["best_chunk"]
                snippets = [t["text"] for t in m.get("top_candidates", []) if t.get("text")]
                if not snippets and best_chunk:
                    snippets = [best_chunk.text]
                verdict = judge_evidence(m["requirement_text"], snippets, llm)
                score = compute_requirement_score(
                    verdict=verdict.verdict,
                    semantic_similarity=m["semantic_similarity"],
                    location_weight=(
                        compute_evidence_location_weight(best_chunk.chunk_type) if best_chunk else 0.40
                    ),
                    recency_weight=compute_recency_weight(best_chunk.metadata) if best_chunk else 0.75,
                    duration_weight=1.0,
                )
                (must_scores if m["requirement_type"] == "must_have" else nice_scores).append(score)

            # Only requirement evidence is available for raw benchmark text (no parsed
            # experience/education), so those two components get zero weight.
            overall = compute_overall_score(
                must_have_score=sum(must_scores) / len(must_scores) if must_scores else 1.0,
                nice_to_have_score=sum(nice_scores) / len(nice_scores) if nice_scores else 1.0,
                experience_fit_score=0.0,
                education_fit_score=0.0,
                weights={"must_have": 0.40, "nice_to_have": 0.20, "experience": 0.0, "education": 0.0},
            )
            full_pool.append({"cand": c, "score": overall})

        full_pool.sort(key=lambda x: x["score"], reverse=True)
        full_relevances = [x["cand"]["ground_truth_relevance"] for x in full_pool]

        # Compute benchmark metrics
        k = min(5, len(candidates))
        k10 = min(10, len(candidates))

        metrics = [
            {
                "metric": f"NDCG@{k}",
                "baseline_ats": compute_ndcg(ats_relevances, k),
                "embeddings_only": compute_ndcg(emb_relevances, k),
                "full_system": compute_ndcg(full_relevances, k),
            },
        ]
        if k10 != k:
            metrics.append({
                "metric": f"NDCG@{k10}",
                "baseline_ats": compute_ndcg(ats_relevances, k10),
                "embeddings_only": compute_ndcg(emb_relevances, k10),
                "full_system": compute_ndcg(full_relevances, k10),
            })
        metrics += [
            {
                "metric": f"Precision@{k}",
                "baseline_ats": compute_precision_at_k(ats_relevances, k),
                "embeddings_only": compute_precision_at_k(emb_relevances, k),
                "full_system": compute_precision_at_k(full_relevances, k),
            },
            {
                "metric": "MRR",
                "baseline_ats": compute_mrr(ats_relevances),
                "embeddings_only": compute_mrr(emb_relevances),
                "full_system": compute_mrr(full_relevances),
            },
        ]

        # Collect keyword trap examples showing where baseline failed
        keyword_traps = []
        for trap in candidates:
            if trap.get("is_keyword_trap"):
                # Find rank in ATS vs Full System
                ats_rank = next((r["rank"] for r in ats_results if r["candidate_id"] == trap["id"]), None)
                full_rank = next((idx + 1 for idx, x in enumerate(full_pool) if x["cand"]["id"] == trap["id"]), None)
                keyword_traps.append({
                    "candidate_name": trap["name"],
                    "description": "Applicant stuffing buzzwords without real production experience.",
                    "baseline_ats_rank": ats_rank,
                    "full_system_rank": full_rank,
                    "behavior": (
                        f"Baseline ATS ranked it #{ats_rank}; "
                        f"RankWise Full System ranked it #{full_rank} of {len(candidates)} "
                        + (
                            "(pushed down by evidence-based grounding)."
                            if (full_rank or 0) > (ats_rank or 0)
                            else "(no improvement over baseline)."
                        )
                    ),
                })

        full_n, ats_n = metrics[0]["full_system"], metrics[0]["baseline_ats"]
        if full_n > ats_n:
            verdict_txt = "outperforms"
        elif full_n < ats_n:
            verdict_txt = "underperforms"
        else:
            verdict_txt = "ties with"
        summary = (
            f"Evaluation on '{role_data['title']}' ({len(candidates)} resumes): Full System "
            f"{verdict_txt} baseline ATS on {metrics[0]['metric']} ({full_n:.3f} vs {ats_n:.3f}); "
            f"{len(keyword_traps)} keyword-trap resume(s) included."
        )

        return {
            "job_role": role_data["title"],
            "metrics": metrics,
            "keyword_trap_examples": keyword_traps,
            "summary": summary,
        }
