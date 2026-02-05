#!/usr/bin/env python3
"""
Phase 3: Embedding & Alignment
==============================

Embeds sentences using kazembed-v5 or LaBSE and aligns parallel sentences
using document-level FAISS indices.

Usage:
    python scripts/03_embed_and_align.py \
        --input_dir chunked_sentences \
        --output_file aligned_corpus.jsonl \
        --model kazembed \
        --similarity_threshold 0.85

Models:
    - kazembed: Nurlykhan/kazembed-v5 (specialized for Kazakh)
    - labse: sentence-transformers/LaBSE (multilingual)

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


# Model configurations
MODELS = {
    "kazembed": "Nurlykhan/kazembed-v5",
    "labse": "sentence-transformers/LaBSE",
}

# Default thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_MIN_WORDS = 5
DEFAULT_LENGTH_RATIO_MIN = 0.5
DEFAULT_LENGTH_RATIO_MAX = 2.0


@dataclass
class AlignmentConfig:
    """Configuration for alignment process."""
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    min_words: int = DEFAULT_MIN_WORDS
    length_ratio_min: float = DEFAULT_LENGTH_RATIO_MIN
    length_ratio_max: float = DEFAULT_LENGTH_RATIO_MAX
    batch_size: int = 64
    use_gpu: bool = True


def load_sentences(file_path: Path) -> List[dict]:
    """Load sentences from JSONL file."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(json.loads(line))
    return sentences


def group_by_document(sentences: List[dict]) -> Dict[str, List[dict]]:
    """
    Group sentences by source document.
    Returns dict: document_id -> list of sentences
    """
    grouped = defaultdict(list)
    for sent in sentences:
        # Extract document ID from source_file
        # e.g., "2024_Annual_Report_KAZ.md" -> "2024_Annual_Report"
        source = sent.get("source_file", "unknown")
        # Remove language suffix
        doc_id = source.rsplit('_', 1)[0]  # Remove "_KAZ.md" etc.
        doc_id = doc_id.replace('.md', '')
        grouped[doc_id].append(sent)
    return dict(grouped)


def load_model(model_name: str, use_gpu: bool = True) -> SentenceTransformer:
    """Load sentence transformer model."""
    model_path = MODELS.get(model_name, model_name)
    print(f"Loading model: {model_path}")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_path, device=device)
    return model


def compute_embeddings(
    model: SentenceTransformer,
    sentences: List[dict],
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute embeddings for a list of sentences.
    Returns numpy array of shape (n_sentences, embedding_dim).
    """
    texts = [s["text"] for s in sentences]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # For cosine similarity via dot product
    )

    return embeddings.astype('float32')


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS index for inner product (cosine similarity with normalized vectors).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def check_length_ratio(sent1: str, sent2: str, min_ratio: float, max_ratio: float) -> bool:
    """Check if length ratio between sentences is within acceptable range."""
    len1 = len(sent1.split())
    len2 = len(sent2.split())

    if len2 == 0:
        return False

    ratio = len1 / len2
    return min_ratio <= ratio <= max_ratio


def check_min_words(sentence: str, min_words: int) -> bool:
    """Check if sentence has minimum number of words."""
    return len(sentence.split()) >= min_words


def align_document_pair(
    kaz_sentences: List[dict],
    kaz_embeddings: np.ndarray,
    target_sentences: List[dict],
    target_embeddings: np.ndarray,
    config: AlignmentConfig
) -> List[Tuple[int, int, float]]:
    """
    Align Kazakh sentences with target language sentences.
    Uses document-level FAISS index for efficient search.

    Returns list of tuples: (kaz_idx, target_idx, similarity_score)
    """
    if len(kaz_sentences) == 0 or len(target_sentences) == 0:
        return []

    # Build FAISS index for target sentences
    index = build_faiss_index(target_embeddings)

    # Search for each Kazakh sentence
    k = 1  # Top-1 match
    distances, indices = index.search(kaz_embeddings, k)

    # Collect matches that pass all filters
    alignments = []
    used_target_indices = set()  # For unique matching

    for kaz_idx in range(len(kaz_sentences)):
        target_idx = indices[kaz_idx][0]
        similarity = distances[kaz_idx][0]

        # Filter 1: Similarity threshold
        if similarity < config.similarity_threshold:
            continue

        # Filter 2: Unique matching (1:1)
        if target_idx in used_target_indices:
            continue

        kaz_text = kaz_sentences[kaz_idx]["text"]
        target_text = target_sentences[target_idx]["text"]

        # Filter 3: Minimum words
        if not check_min_words(kaz_text, config.min_words):
            continue
        if not check_min_words(target_text, config.min_words):
            continue

        # Filter 4: Length ratio
        if not check_length_ratio(
            kaz_text, target_text,
            config.length_ratio_min, config.length_ratio_max
        ):
            continue

        # All filters passed
        alignments.append((kaz_idx, target_idx, float(similarity)))
        used_target_indices.add(target_idx)

    return alignments


def create_aligned_triplet(
    kaz_sent: dict,
    rus_sent: Optional[dict],
    eng_sent: Optional[dict],
    sim_kaz_rus: float,
    sim_kaz_eng: float,
    doc_id: str,
    triplet_idx: int
) -> dict:
    """Create aligned triplet dictionary."""
    triplet = {
        "id": f"{doc_id}_{triplet_idx:04d}",
        "kaz": kaz_sent["text"],
        "rus": rus_sent["text"] if rus_sent else None,
        "eng": eng_sent["text"] if eng_sent else None,
        "similarity_kaz_rus": round(sim_kaz_rus, 4) if rus_sent else None,
        "similarity_kaz_eng": round(sim_kaz_eng, 4) if eng_sent else None,
        "metadata": {
            "source_doc": doc_id,
            "kaz_idx": kaz_sent.get("sentence_idx"),
            "year": kaz_sent.get("metadata", {}).get("year"),
            "company": kaz_sent.get("metadata", {}).get("company"),
            "doc_type": kaz_sent.get("metadata", {}).get("type")
        }
    }
    return triplet


def run_ab_test(
    model_kazembed: SentenceTransformer,
    model_labse: SentenceTransformer,
    kaz_sentences: List[dict],
    rus_sentences: List[dict],
    sample_size: int = 100
) -> dict:
    """
    Run A/B test comparing kazembed-v5 vs LaBSE for cross-lingual alignment.
    """
    print("\n" + "=" * 60)
    print("Running A/B Test: kazembed-v5 vs LaBSE")
    print("=" * 60)

    # Sample sentences
    sample_kaz = kaz_sentences[:min(sample_size, len(kaz_sentences))]
    sample_rus = rus_sentences[:min(sample_size, len(rus_sentences))]

    results = {}

    for model_name, model in [("kazembed", model_kazembed), ("labse", model_labse)]:
        print(f"\nTesting {model_name}...")

        # Compute embeddings
        kaz_emb = compute_embeddings(model, sample_kaz, show_progress=False)
        rus_emb = compute_embeddings(model, sample_rus, show_progress=False)

        # Build index and search
        index = build_faiss_index(rus_emb)
        distances, indices = index.search(kaz_emb, k=1)

        # Calculate metrics
        similarities = distances[:, 0]
        results[model_name] = {
            "mean_similarity": float(np.mean(similarities)),
            "median_similarity": float(np.median(similarities)),
            "std_similarity": float(np.std(similarities)),
            "above_0.8": int(np.sum(similarities > 0.8)),
            "above_0.85": int(np.sum(similarities > 0.85)),
            "above_0.9": int(np.sum(similarities > 0.9))
        }

        print(f"  Mean similarity: {results[model_name]['mean_similarity']:.4f}")
        print(f"  Pairs > 0.85: {results[model_name]['above_0.85']}/{len(sample_kaz)}")

    # Determine winner
    kazembed_score = results["kazembed"]["above_0.85"]
    labse_score = results["labse"]["above_0.85"]

    winner = "kazembed" if kazembed_score >= labse_score else "labse"
    print(f"\nRecommended model: {winner}")

    return {
        "results": results,
        "winner": winner,
        "sample_size": len(sample_kaz)
    }


def main():
    parser = argparse.ArgumentParser(description="Embed and align parallel sentences")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="chunked_sentences",
        help="Input directory with sentence JSONL files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="aligned_corpus.jsonl",
        help="Output file for aligned corpus"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["kazembed", "labse", "auto"],
        default="auto",
        help="Embedding model to use (auto runs A/B test first)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Minimum similarity score for alignment"
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help="Minimum words per sentence"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU usage"
    )
    parser.add_argument(
        "--ab_test_only",
        action="store_true",
        help="Only run A/B test, don't perform alignment"
    )
    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_file = base_dir / args.output_file

    config = AlignmentConfig(
        similarity_threshold=args.similarity_threshold,
        min_words=args.min_words,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )

    # Load sentences
    print("Loading sentences...")
    kaz_file = input_dir / "sentences_kaz.jsonl"
    rus_file = input_dir / "sentences_rus.jsonl"
    eng_file = input_dir / "sentences_eng.jsonl"

    kaz_sentences = load_sentences(kaz_file) if kaz_file.exists() else []
    rus_sentences = load_sentences(rus_file) if rus_file.exists() else []
    eng_sentences = load_sentences(eng_file) if eng_file.exists() else []

    print(f"  KAZ: {len(kaz_sentences):,} sentences")
    print(f"  RUS: {len(rus_sentences):,} sentences")
    print(f"  ENG: {len(eng_sentences):,} sentences")

    if len(kaz_sentences) == 0:
        print("[ERROR] No Kazakh sentences found. Cannot proceed.")
        return

    # Determine model to use
    model_name = args.model
    ab_test_results = None

    if model_name == "auto" or args.ab_test_only:
        # Run A/B test
        print("\nLoading both models for A/B test...")
        model_kazembed = load_model("kazembed", config.use_gpu)
        model_labse = load_model("labse", config.use_gpu)

        ab_test_results = run_ab_test(
            model_kazembed, model_labse,
            kaz_sentences, rus_sentences if rus_sentences else eng_sentences
        )

        if args.ab_test_only:
            # Save A/B test results and exit
            ab_results_file = base_dir / "ab_test_results.json"
            with open(ab_results_file, 'w', encoding='utf-8') as f:
                json.dump(ab_test_results, f, indent=2)
            print(f"\nA/B test results saved to: {ab_results_file}")
            return

        model_name = ab_test_results["winner"]
        model = model_kazembed if model_name == "kazembed" else model_labse
    else:
        model = load_model(model_name, config.use_gpu)

    print(f"\nUsing model: {model_name}")

    # Group sentences by document
    print("\nGrouping sentences by document...")
    kaz_by_doc = group_by_document(kaz_sentences)
    rus_by_doc = group_by_document(rus_sentences)
    eng_by_doc = group_by_document(eng_sentences)

    print(f"  KAZ documents: {len(kaz_by_doc)}")
    print(f"  RUS documents: {len(rus_by_doc)}")
    print(f"  ENG documents: {len(eng_by_doc)}")

    # Find common documents
    all_docs = set(kaz_by_doc.keys())
    docs_with_rus = all_docs & set(rus_by_doc.keys())
    docs_with_eng = all_docs & set(eng_by_doc.keys())

    print(f"\n  Documents with KAZ+RUS: {len(docs_with_rus)}")
    print(f"  Documents with KAZ+ENG: {len(docs_with_eng)}")

    # Alignment
    print("\n" + "=" * 60)
    print("Starting alignment process...")
    print("=" * 60)

    all_triplets = []
    alignment_stats = {
        "total_kaz_sentences": len(kaz_sentences),
        "aligned_kaz_rus": 0,
        "aligned_kaz_eng": 0,
        "full_triplets": 0,
        "partial_triplets": 0
    }

    for doc_id in tqdm(kaz_by_doc.keys(), desc="Processing documents"):
        kaz_sents = kaz_by_doc[doc_id]

        # Compute Kazakh embeddings
        kaz_emb = compute_embeddings(model, kaz_sents, config.batch_size, show_progress=False)

        # Align with Russian
        kaz_rus_alignments = {}
        if doc_id in rus_by_doc:
            rus_sents = rus_by_doc[doc_id]
            rus_emb = compute_embeddings(model, rus_sents, config.batch_size, show_progress=False)
            alignments = align_document_pair(kaz_sents, kaz_emb, rus_sents, rus_emb, config)
            kaz_rus_alignments = {kaz_idx: (rus_idx, sim) for kaz_idx, rus_idx, sim in alignments}
            alignment_stats["aligned_kaz_rus"] += len(alignments)

        # Align with English
        kaz_eng_alignments = {}
        if doc_id in eng_by_doc:
            eng_sents = eng_by_doc[doc_id]
            eng_emb = compute_embeddings(model, eng_sents, config.batch_size, show_progress=False)
            alignments = align_document_pair(kaz_sents, kaz_emb, eng_sents, eng_emb, config)
            kaz_eng_alignments = {kaz_idx: (eng_idx, sim) for kaz_idx, eng_idx, sim in alignments}
            alignment_stats["aligned_kaz_eng"] += len(alignments)

        # Create triplets
        triplet_idx = 0
        for kaz_idx, kaz_sent in enumerate(kaz_sents):
            rus_match = kaz_rus_alignments.get(kaz_idx)
            eng_match = kaz_eng_alignments.get(kaz_idx)

            # Skip if no matches
            if rus_match is None and eng_match is None:
                continue

            rus_sent = rus_by_doc[doc_id][rus_match[0]] if rus_match else None
            eng_sent = eng_by_doc[doc_id][eng_match[0]] if eng_match else None
            sim_kaz_rus = rus_match[1] if rus_match else 0.0
            sim_kaz_eng = eng_match[1] if eng_match else 0.0

            triplet = create_aligned_triplet(
                kaz_sent, rus_sent, eng_sent,
                sim_kaz_rus, sim_kaz_eng,
                doc_id, triplet_idx
            )
            all_triplets.append(triplet)
            triplet_idx += 1

            # Update stats
            if rus_sent and eng_sent:
                alignment_stats["full_triplets"] += 1
            else:
                alignment_stats["partial_triplets"] += 1

    # Save aligned corpus
    print(f"\nSaving {len(all_triplets):,} aligned triplets...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in all_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    # Save alignment metadata
    metadata_file = output_file.with_suffix('.meta.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "config": {
                "similarity_threshold": config.similarity_threshold,
                "min_words": config.min_words,
                "length_ratio_min": config.length_ratio_min,
                "length_ratio_max": config.length_ratio_max
            },
            "stats": alignment_stats,
            "ab_test": ab_test_results
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"  Total KAZ sentences:  {alignment_stats['total_kaz_sentences']:,}")
    print(f"  Aligned KAZ-RUS:      {alignment_stats['aligned_kaz_rus']:,}")
    print(f"  Aligned KAZ-ENG:      {alignment_stats['aligned_kaz_eng']:,}")
    print(f"  Full triplets:        {alignment_stats['full_triplets']:,}")
    print(f"  Partial triplets:     {alignment_stats['partial_triplets']:,}")
    print(f"  Total aligned:        {len(all_triplets):,}")
    print(f"\n  Output file: {output_file.absolute()}")
    print(f"  Metadata:    {metadata_file.absolute()}")


if __name__ == "__main__":
    main()
