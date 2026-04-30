from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline PepCLIP topK retrieval from exported embeddings.")
    parser.add_argument("--query_embeddings", required=True, help="Usually receptor_embeddings.npy")
    parser.add_argument("--target_embeddings", required=True, help="Usually peptide_embeddings.npy")
    parser.add_argument(
        "--metadata_jsonl",
        default=None,
        help="Shared metadata for paired retrieval. Kept for backward compatibility.",
    )
    parser.add_argument("--query_metadata_jsonl", default=None)
    parser.add_argument("--target_metadata_jsonl", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument(
        "--paired",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute paired metrics by treating query row i and target row i as the true pair.",
    )
    return parser.parse_args()


def read_metadata(path: str | Path) -> List[Dict[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def topk_numpy(query: np.ndarray, target: np.ndarray, top_k: int, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    all_scores = []
    all_indices = []
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        scores = query[start:end] @ target.T
        kth = min(top_k, target.shape[0]) - 1
        part = np.argpartition(-scores, kth=kth, axis=1)[:, : top_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        all_indices.append(np.take_along_axis(part, order, axis=1))
        all_scores.append(np.take_along_axis(part_scores, order, axis=1))
    return np.concatenate(all_scores, axis=0), np.concatenate(all_indices, axis=0)


def topk_faiss(query: np.ndarray, target: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    import faiss

    index = faiss.IndexFlatIP(target.shape[1])
    index.add(target.astype(np.float32, copy=False))
    scores, indices = index.search(query.astype(np.float32, copy=False), top_k)
    return scores, indices


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import faiss  # noqa: F401
    except ImportError:
        return "numpy"
    return "faiss"


def main() -> None:
    args = parse_args()
    query = np.load(args.query_embeddings).astype(np.float32, copy=False)
    target = np.load(args.target_embeddings).astype(np.float32, copy=False)
    query_metadata_path = args.query_metadata_jsonl or args.metadata_jsonl
    target_metadata_path = args.target_metadata_jsonl or args.metadata_jsonl
    if query_metadata_path is None or target_metadata_path is None:
        raise ValueError("Provide --metadata_jsonl or both --query_metadata_jsonl and --target_metadata_jsonl")
    query_metadata = read_metadata(query_metadata_path)
    target_metadata = read_metadata(target_metadata_path)
    if query.shape[0] != len(query_metadata):
        raise ValueError("Query embedding row count must match query metadata row count")
    if target.shape[0] != len(target_metadata):
        raise ValueError("Target embedding row count must match target metadata row count")

    paired = query.shape[0] == target.shape[0] if args.paired is None else args.paired
    if paired and query.shape[0] != target.shape[0]:
        raise ValueError("Paired retrieval requires query and target row counts to match")

    backend = choose_backend(args.backend)
    if backend == "faiss":
        scores, indices = topk_faiss(query, target, args.top_k)
    else:
        scores, indices = topk_numpy(query, target, args.top_k, args.chunk_size)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    hit_reciprocal_ranks = []
    with output_path.open("w", encoding="utf-8") as handle:
        for i in range(query.shape[0]):
            retrieved = []
            rank = None
            for j, target_idx in enumerate(indices[i].tolist()):
                target_row = target_metadata[target_idx]
                if paired and target_idx == i:
                    rank = j + 1
                retrieved.append(
                    {
                        "rank": j + 1,
                        "target_index": int(target_idx),
                        "score": float(scores[i, j]),
                        "sample_id": target_row["sample_id"],
                        "peptide_sequence": target_row.get("peptide_sequence", target_row.get("peptide_key", "")),
                    }
                )
            if paired:
                reciprocal_rank = 1.0 / rank if rank is not None else 0.0
                reciprocal_ranks.append(reciprocal_rank)
            if paired and rank is not None:
                hit_reciprocal_ranks.append(1.0 / rank)
                hits_at_1 += int(rank <= 1)
                hits_at_5 += int(rank <= 5)
                hits_at_10 += int(rank <= 10)
            row = {
                "query_index": i,
                "query_sample_id": query_metadata[i]["sample_id"],
                "query_receptor_key": query_metadata[i]["receptor_key"],
                "retrieved": retrieved,
            }
            if paired:
                row.update(
                    {
                        "true_peptide_sequence": query_metadata[i].get(
                            "peptide_sequence",
                            query_metadata[i].get("peptide_key", ""),
                        ),
                        "true_peptide_key": query_metadata[i].get("peptide_key", ""),
                        "true_rank_within_top_k": rank,
                    }
                )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = query.shape[0]
    summary = {
        "backend": backend,
        "num_queries": n,
        "num_targets": int(target.shape[0]),
        "top_k": args.top_k,
        "paired": paired,
        "output_jsonl": str(output_path),
    }
    if paired:
        summary.update(
            {
                "recall_at_1": hits_at_1 / n,
                "recall_at_5": hits_at_5 / n,
                "recall_at_10": hits_at_10 / n,
                "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
                "hit_mrr_within_top_k": float(np.mean(hit_reciprocal_ranks)) if hit_reciprocal_ranks else 0.0,
            }
        )
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
