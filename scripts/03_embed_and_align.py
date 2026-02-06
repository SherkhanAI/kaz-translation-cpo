#!/usr/bin/env python3
"""
Phase 3: Embedding & Alignment with Bertalign + MEXMA
=====================================================

Aligns parallel sentences using Bertalign (m:n dynamic programming alignment)
with MEXMA embeddings (SOTA cross-lingual, outperforms SONAR on xsim++).

Supports many-to-many alignments (e.g., 2 Kazakh sentences = 1 Russian sentence).

Models:
    - MEXMA: facebook/MEXMA via SentenceTransformer (default, SOTA)
    - SONAR: text_sonar_basic_encoder (fallback, FAISS 1:1 only)
    - LaBSE: sentence-transformers/LaBSE (fallback)

Usage:
    python scripts/03_embed_and_align.py \
        --input_dir chunked_sentences \
        --output_file aligned_corpus.jsonl \
        --model mexma \
        --similarity_threshold 0.85

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import torch

# Try to import SONAR (for --model sonar and --compare_only)
SONAR_AVAILABLE = False
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    SONAR_AVAILABLE = True
except ImportError:
    pass

# sentence-transformers (for MEXMA and LaBSE)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Bertalign
BERTALIGN_AVAILABLE = False
try:
    import bertalign
    from bertalign.encoder import Encoder as BertalignEncoder
    from bertalign import Bertalign
    BERTALIGN_AVAILABLE = True
except ImportError:
    pass

# FAISS (for SONAR fallback path)
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass


# SONAR language codes
SONAR_LANG_CODES = {
    "kaz": "kaz_Cyrl",
    "rus": "rus_Cyrl",
    "eng": "eng_Latn",
}

# Default thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_MIN_WORDS = 5
DEFAULT_LENGTH_RATIO_MIN = 0.3
DEFAULT_LENGTH_RATIO_MAX = 3.0


@dataclass
class AlignmentConfig:
    """Configuration for alignment process."""
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    min_words: int = DEFAULT_MIN_WORDS
    length_ratio_min: float = DEFAULT_LENGTH_RATIO_MIN
    length_ratio_max: float = DEFAULT_LENGTH_RATIO_MAX
    batch_size: int = 32
    use_gpu: bool = True
    max_align: int = 5
    top_k: int = 3


class SONAREncoder:
    """Wrapper for SONAR text encoder."""

    def __init__(self, device: str = "cuda"):
        if not SONAR_AVAILABLE:
            raise RuntimeError("SONAR not available. Install with: pip install sonar-space")

        self.device = device
        print(f"Loading SONAR encoder on {device}...")

        self.model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder"
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model.model = self.model.model.to(device)

    def encode(
        self,
        texts: List[str],
        lang: str,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts using SONAR. Returns normalized float32 embeddings."""
        lang_code = SONAR_LANG_CODES.get(lang, lang)

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Encoding {lang}")

        for i in iterator:
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.model.predict(batch, source_lang=lang_code)
                if hasattr(embeddings, 'cpu'):
                    embeddings = embeddings.cpu().numpy()
                elif not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings)
                all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings).astype('float32')
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / (norms + 1e-8)
        return result


class SentenceTransformerEncoder:
    """Wrapper for SentenceTransformer models (MEXMA, LaBSE)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not available")

        self.model_name = model_name
        print(f"Loading {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: List[str],
        lang: str = "",
        batch_size: int = 64,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts. Lang parameter is ignored (model is language-agnostic)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype('float32')


# Model name mapping
MODEL_NAMES = {
    "mexma": "facebook/MEXMA",
    "labse": "sentence-transformers/LaBSE",
}


def load_sentences(file_path: Path) -> List[dict]:
    """Load sentences from JSONL file."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(json.loads(line))
    return sentences


def group_by_document(sentences: List[dict]) -> Dict[str, List[dict]]:
    """Group sentences by source document."""
    grouped = defaultdict(list)
    for sent in sentences:
        source = sent.get("source_file", "unknown")
        doc_id = source.rsplit('_', 1)[0]
        doc_id = doc_id.replace('.md', '')
        grouped[doc_id].append(sent)
    return dict(grouped)


def inject_bertalign_model(model_key: str, device: str = "cuda"):
    """
    Inject a custom SentenceTransformer model into Bertalign's global state.
    Bertalign uses a module-level `model` global loaded at import time.
    """
    if not BERTALIGN_AVAILABLE:
        raise RuntimeError("bertalign not available. Install with: pip install bertalign")

    hf_name = MODEL_NAMES[model_key]
    print(f"Injecting {hf_name} into Bertalign...")

    encoder = BertalignEncoder(hf_name)
    # Move to correct device
    encoder.model = SentenceTransformer(hf_name, device=device)

    # Patch the module-level globals that Bertalign's aligner reads
    bertalign.model = encoder
    import bertalign.aligner
    bertalign.aligner.model = encoder

    return encoder


def align_document_bertalign(
    src_sentences: List[dict],
    tgt_sentences: List[dict],
    config: AlignmentConfig,
) -> List[Tuple[List[int], List[int]]]:
    """
    Align sentences using Bertalign's m:n DP algorithm.
    Returns list of (src_indices, tgt_indices) beads.
    """
    if len(src_sentences) == 0 or len(tgt_sentences) == 0:
        return []

    src_texts = [s["text"] for s in src_sentences]
    tgt_texts = [s["text"] for s in tgt_sentences]

    src_text = "\n".join(src_texts)
    tgt_text = "\n".join(tgt_texts)

    aligner = Bertalign(
        src_text,
        tgt_text,
        is_split=True,
        max_align=config.max_align,
        top_k=config.top_k,
    )
    aligner.align_sents()

    return aligner.result


def compute_post_hoc_similarity(
    encoder,
    src_text: str,
    tgt_text: str,
) -> float:
    """Compute cosine similarity between two texts using the encoder."""
    emb = encoder.encode([src_text, tgt_text])
    sim = float(np.dot(emb[0], emb[1]))
    return sim


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index for inner product search."""
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss not available. Install with: pip install faiss-cpu")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def align_document_faiss(
    kaz_sentences: List[dict],
    kaz_embeddings: np.ndarray,
    target_sentences: List[dict],
    target_embeddings: np.ndarray,
    config: AlignmentConfig
) -> List[Tuple[int, int, float]]:
    """
    SONAR fallback: FAISS top-1 nearest-neighbor (1:1 only).
    Returns list of (kaz_idx, target_idx, similarity).
    """
    if len(kaz_sentences) == 0 or len(target_sentences) == 0:
        return []

    index = build_faiss_index(target_embeddings)
    distances, indices = index.search(kaz_embeddings, 1)

    alignments = []
    used_target_indices = set()

    for kaz_idx in range(len(kaz_sentences)):
        target_idx = indices[kaz_idx][0]
        similarity = distances[kaz_idx][0]

        if similarity < config.similarity_threshold:
            continue
        if target_idx in used_target_indices:
            continue

        kaz_text = kaz_sentences[kaz_idx]["text"]
        target_text = target_sentences[target_idx]["text"]

        if len(kaz_text.split()) < config.min_words:
            continue
        if len(target_text.split()) < config.min_words:
            continue

        len_kaz = len(kaz_text.split())
        len_tgt = len(target_text.split())
        if len_tgt > 0:
            ratio = len_kaz / len_tgt
            if not (config.length_ratio_min <= ratio <= config.length_ratio_max):
                continue

        alignments.append((kaz_idx, target_idx, float(similarity)))
        used_target_indices.add(target_idx)

    return alignments


def create_aligned_triplet(
    kaz_text: str,
    rus_text: Optional[str],
    eng_text: Optional[str],
    sim_kaz_rus: float,
    sim_kaz_eng: float,
    alignment_type_kaz_rus: str,
    alignment_type_kaz_eng: str,
    doc_id: str,
    triplet_idx: int,
    metadata_base: dict,
) -> dict:
    """Create aligned triplet dictionary with m:n alignment info."""
    triplet = {
        "id": f"{doc_id}_{triplet_idx:04d}",
        "kaz": kaz_text,
        "rus": rus_text,
        "eng": eng_text,
        "similarity_kaz_rus": round(sim_kaz_rus, 4) if rus_text else None,
        "similarity_kaz_eng": round(sim_kaz_eng, 4) if eng_text else None,
        "alignment_type_kaz_rus": alignment_type_kaz_rus if rus_text else None,
        "alignment_type_kaz_eng": alignment_type_kaz_eng if eng_text else None,
        "metadata": metadata_base,
    }
    return triplet


def process_bertalign_beads(
    beads: List[Tuple[List[int], List[int]]],
    src_sentences: List[dict],
    tgt_sentences: List[dict],
    encoder,
    config: AlignmentConfig,
) -> List[Tuple[str, str, float, str]]:
    """
    Process Bertalign beads into (src_text, tgt_text, similarity, alignment_type) tuples.
    Filters by min_words, length_ratio, and similarity_threshold.
    """
    results = []

    for src_indices, tgt_indices in beads:
        # Skip empty alignments (deletions/insertions)
        if not src_indices or not tgt_indices:
            continue

        src_texts = [src_sentences[i]["text"] for i in src_indices]
        tgt_texts = [tgt_sentences[i]["text"] for i in tgt_indices]

        src_concat = " ".join(src_texts)
        tgt_concat = " ".join(tgt_texts)

        # Min words filter
        if len(src_concat.split()) < config.min_words:
            continue
        if len(tgt_concat.split()) < config.min_words:
            continue

        # Length ratio filter
        len_src = len(src_concat.split())
        len_tgt = len(tgt_concat.split())
        if len_tgt > 0:
            ratio = len_src / len_tgt
            if not (config.length_ratio_min <= ratio <= config.length_ratio_max):
                continue

        # Compute post-hoc similarity
        similarity = compute_post_hoc_similarity(encoder, src_concat, tgt_concat)

        if similarity < config.similarity_threshold:
            continue

        alignment_type = f"{len(src_indices)}:{len(tgt_indices)}"
        results.append((src_concat, tgt_concat, similarity, alignment_type))

    return results


def compare_models(
    kaz_sentences: List[dict],
    rus_sentences: List[dict],
    sample_size: int = 100,
    device: str = "cuda"
) -> dict:
    """Compare MEXMA vs SONAR vs LaBSE for cross-lingual alignment quality."""
    print("\n" + "=" * 60)
    print("Comparing MEXMA vs SONAR vs LaBSE")
    print("=" * 60)

    sample_kaz = kaz_sentences[:min(sample_size, len(kaz_sentences))]
    sample_rus = rus_sentences[:min(sample_size, len(rus_sentences))]

    kaz_texts = [s["text"] for s in sample_kaz]
    rus_texts = [s["text"] for s in sample_rus]

    results = {}

    # Test MEXMA
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\nTesting MEXMA (facebook/MEXMA)...")
        try:
            enc = SentenceTransformerEncoder("facebook/MEXMA", device=device)
            kaz_emb = enc.encode(kaz_texts, "kaz")
            rus_emb = enc.encode(rus_texts, "rus")

            index = build_faiss_index(rus_emb)
            distances, _ = index.search(kaz_emb, k=1)
            similarities = distances[:, 0]

            results["mexma"] = {
                "mean_similarity": float(np.mean(similarities)),
                "median_similarity": float(np.median(similarities)),
                "std_similarity": float(np.std(similarities)),
                "above_0.8": int(np.sum(similarities > 0.8)),
                "above_0.85": int(np.sum(similarities > 0.85)),
                "above_0.9": int(np.sum(similarities > 0.9))
            }
            print(f"  Mean similarity: {results['mexma']['mean_similarity']:.4f}")
            print(f"  Pairs > 0.85: {results['mexma']['above_0.85']}/{len(sample_kaz)}")
            del enc
        except Exception as e:
            print(f"  MEXMA error: {e}")

    # Test SONAR
    if SONAR_AVAILABLE:
        print("\nTesting SONAR...")
        try:
            sonar = SONAREncoder(device=device)
            kaz_emb = sonar.encode(kaz_texts, "kaz")
            rus_emb = sonar.encode(rus_texts, "rus")

            index = build_faiss_index(rus_emb)
            distances, _ = index.search(kaz_emb, k=1)
            similarities = distances[:, 0]

            results["sonar"] = {
                "mean_similarity": float(np.mean(similarities)),
                "median_similarity": float(np.median(similarities)),
                "std_similarity": float(np.std(similarities)),
                "above_0.8": int(np.sum(similarities > 0.8)),
                "above_0.85": int(np.sum(similarities > 0.85)),
                "above_0.9": int(np.sum(similarities > 0.9))
            }
            print(f"  Mean similarity: {results['sonar']['mean_similarity']:.4f}")
            print(f"  Pairs > 0.85: {results['sonar']['above_0.85']}/{len(sample_kaz)}")
            del sonar
        except Exception as e:
            print(f"  SONAR error: {e}")

    # Test LaBSE
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\nTesting LaBSE...")
        try:
            labse = SentenceTransformerEncoder("sentence-transformers/LaBSE", device=device)
            kaz_emb = labse.encode(kaz_texts, "kaz")
            rus_emb = labse.encode(rus_texts, "rus")

            index = build_faiss_index(rus_emb)
            distances, _ = index.search(kaz_emb, k=1)
            similarities = distances[:, 0]

            results["labse"] = {
                "mean_similarity": float(np.mean(similarities)),
                "median_similarity": float(np.median(similarities)),
                "std_similarity": float(np.std(similarities)),
                "above_0.8": int(np.sum(similarities > 0.8)),
                "above_0.85": int(np.sum(similarities > 0.85)),
                "above_0.9": int(np.sum(similarities > 0.9))
            }
            print(f"  Mean similarity: {results['labse']['mean_similarity']:.4f}")
            print(f"  Pairs > 0.85: {results['labse']['above_0.85']}/{len(sample_kaz)}")
            del labse
        except Exception as e:
            print(f"  LaBSE error: {e}")

    # Determine winner by pairs above 0.85
    winner = "mexma"
    best_score = -1
    for model_name, model_results in results.items():
        score = model_results.get("above_0.85", 0)
        if score > best_score:
            best_score = score
            winner = model_name

    print(f"\nRecommended model: {winner.upper()}")

    return {
        "results": results,
        "winner": winner,
        "sample_size": len(sample_kaz)
    }


def run_bertalign_pipeline(args, config, device, kaz_sentences, rus_sentences, eng_sentences, base_dir):
    """Main pipeline using Bertalign m:n alignment (for MEXMA or LaBSE)."""

    # Inject model into Bertalign
    encoder = inject_bertalign_model(args.model, device=device)

    # Group sentences by document
    print("\nGrouping sentences by document...")
    kaz_by_doc = group_by_document(kaz_sentences)
    rus_by_doc = group_by_document(rus_sentences)
    eng_by_doc = group_by_document(eng_sentences)

    print(f"  KAZ documents: {len(kaz_by_doc)}")
    print(f"  RUS documents: {len(rus_by_doc)}")
    print(f"  ENG documents: {len(eng_by_doc)}")

    docs_with_rus = set(kaz_by_doc.keys()) & set(rus_by_doc.keys())
    docs_with_eng = set(kaz_by_doc.keys()) & set(eng_by_doc.keys())
    print(f"\n  Documents with KAZ+RUS: {len(docs_with_rus)}")
    print(f"  Documents with KAZ+ENG: {len(docs_with_eng)}")

    # Alignment
    print("\n" + "=" * 60)
    print("Starting Bertalign m:n alignment...")
    print("=" * 60)

    all_triplets = []
    alignment_stats = {
        "total_kaz_sentences": len(kaz_sentences),
        "aligned_kaz_rus": 0,
        "aligned_kaz_eng": 0,
        "full_triplets": 0,
        "partial_triplets": 0,
        "many_to_many_kaz_rus": 0,
        "many_to_many_kaz_eng": 0,
    }

    for doc_id in tqdm(kaz_by_doc.keys(), desc="Processing documents"):
        kaz_sents = kaz_by_doc[doc_id]

        # --- KAZ-RUS alignment ---
        kaz_rus_aligned = []
        if doc_id in rus_by_doc:
            rus_sents = rus_by_doc[doc_id]
            try:
                beads = align_document_bertalign(kaz_sents, rus_sents, config)
                kaz_rus_aligned = process_bertalign_beads(
                    beads, kaz_sents, rus_sents, encoder, config
                )
                alignment_stats["aligned_kaz_rus"] += len(kaz_rus_aligned)
                alignment_stats["many_to_many_kaz_rus"] += sum(
                    1 for _, _, _, at in kaz_rus_aligned if at != "1:1"
                )
            except Exception as e:
                print(f"\n  [WARN] Bertalign failed for {doc_id} KAZ-RUS: {e}")

        # --- KAZ-ENG alignment ---
        kaz_eng_aligned = []
        if doc_id in eng_by_doc:
            eng_sents = eng_by_doc[doc_id]
            try:
                beads = align_document_bertalign(kaz_sents, eng_sents, config)
                kaz_eng_aligned = process_bertalign_beads(
                    beads, kaz_sents, eng_sents, encoder, config
                )
                alignment_stats["aligned_kaz_eng"] += len(kaz_eng_aligned)
                alignment_stats["many_to_many_kaz_eng"] += sum(
                    1 for _, _, _, at in kaz_eng_aligned if at != "1:1"
                )
            except Exception as e:
                print(f"\n  [WARN] Bertalign failed for {doc_id} KAZ-ENG: {e}")

        # Build triplets: merge KAZ-RUS and KAZ-ENG alignments by KAZ text
        # Since Bertalign aligns sequentially, we match by position in kaz_text
        kaz_rus_map = {kaz_text: (rus_text, sim, at) for kaz_text, rus_text, sim, at in kaz_rus_aligned}
        kaz_eng_map = {kaz_text: (eng_text, sim, at) for kaz_text, eng_text, sim, at in kaz_eng_aligned}

        all_kaz_keys = list(dict.fromkeys(
            [k for k, _, _, _ in kaz_rus_aligned] +
            [k for k, _, _, _ in kaz_eng_aligned]
        ))

        triplet_idx = 0
        for kaz_text in all_kaz_keys:
            rus_match = kaz_rus_map.get(kaz_text)
            eng_match = kaz_eng_map.get(kaz_text)

            if rus_match is None and eng_match is None:
                continue

            rus_text = rus_match[0] if rus_match else None
            sim_kaz_rus = rus_match[1] if rus_match else 0.0
            at_kaz_rus = rus_match[2] if rus_match else "0:0"

            eng_text = eng_match[0] if eng_match else None
            sim_kaz_eng = eng_match[1] if eng_match else 0.0
            at_kaz_eng = eng_match[2] if eng_match else "0:0"

            # Extract metadata from first kaz sentence in this document
            metadata_base = {
                "source_doc": doc_id,
                "year": kaz_sents[0].get("metadata", {}).get("year"),
                "company": kaz_sents[0].get("metadata", {}).get("company"),
                "doc_type": kaz_sents[0].get("metadata", {}).get("type"),
            }

            triplet = create_aligned_triplet(
                kaz_text, rus_text, eng_text,
                sim_kaz_rus, sim_kaz_eng,
                at_kaz_rus, at_kaz_eng,
                doc_id, triplet_idx,
                metadata_base,
            )
            all_triplets.append(triplet)
            triplet_idx += 1

            if rus_text and eng_text:
                alignment_stats["full_triplets"] += 1
            else:
                alignment_stats["partial_triplets"] += 1

    return all_triplets, alignment_stats


def run_sonar_pipeline(args, config, device, kaz_sentences, rus_sentences, eng_sentences, base_dir):
    """Fallback pipeline using SONAR + FAISS 1:1 alignment."""

    encoder = SONAREncoder(device=device)

    # Group by document
    print("\nGrouping sentences by document...")
    kaz_by_doc = group_by_document(kaz_sentences)
    rus_by_doc = group_by_document(rus_sentences)
    eng_by_doc = group_by_document(eng_sentences)

    print(f"  KAZ documents: {len(kaz_by_doc)}")
    print(f"  RUS documents: {len(rus_by_doc)}")
    print(f"  ENG documents: {len(eng_by_doc)}")

    docs_with_rus = set(kaz_by_doc.keys()) & set(rus_by_doc.keys())
    docs_with_eng = set(kaz_by_doc.keys()) & set(eng_by_doc.keys())
    print(f"\n  Documents with KAZ+RUS: {len(docs_with_rus)}")
    print(f"  Documents with KAZ+ENG: {len(docs_with_eng)}")

    # Pre-compute all embeddings
    print("\n" + "=" * 60)
    print("Computing SONAR embeddings...")
    print("=" * 60)

    all_kaz_texts = [s["text"] for s in kaz_sentences]
    all_rus_texts = [s["text"] for s in rus_sentences] if rus_sentences else []
    all_eng_texts = [s["text"] for s in eng_sentences] if eng_sentences else []

    print(f"\nEncoding {len(all_kaz_texts):,} Kazakh sentences...")
    all_kaz_emb = encoder.encode(all_kaz_texts, "kaz", config.batch_size, show_progress=True)

    all_rus_emb = None
    if all_rus_texts:
        print(f"Encoding {len(all_rus_texts):,} Russian sentences...")
        all_rus_emb = encoder.encode(all_rus_texts, "rus", config.batch_size, show_progress=True)

    all_eng_emb = None
    if all_eng_texts:
        print(f"Encoding {len(all_eng_texts):,} English sentences...")
        all_eng_emb = encoder.encode(all_eng_texts, "eng", config.batch_size, show_progress=True)

    # Index mappings
    kaz_idx_map = {id(s): i for i, s in enumerate(kaz_sentences)}
    rus_idx_map = {id(s): i for i, s in enumerate(rus_sentences)} if rus_sentences else {}
    eng_idx_map = {id(s): i for i, s in enumerate(eng_sentences)} if eng_sentences else {}

    # Alignment
    print("\n" + "=" * 60)
    print("Starting SONAR FAISS 1:1 alignment...")
    print("=" * 60)

    all_triplets = []
    alignment_stats = {
        "total_kaz_sentences": len(kaz_sentences),
        "aligned_kaz_rus": 0,
        "aligned_kaz_eng": 0,
        "full_triplets": 0,
        "partial_triplets": 0,
        "many_to_many_kaz_rus": 0,
        "many_to_many_kaz_eng": 0,
    }

    for doc_id in tqdm(kaz_by_doc.keys(), desc="Processing documents"):
        kaz_sents = kaz_by_doc[doc_id]
        kaz_indices = [kaz_idx_map[id(s)] for s in kaz_sents]
        kaz_emb = all_kaz_emb[kaz_indices]

        # Align with Russian
        kaz_rus_alignments = {}
        if doc_id in rus_by_doc:
            rus_sents = rus_by_doc[doc_id]
            rus_indices = [rus_idx_map[id(s)] for s in rus_sents]
            rus_emb = all_rus_emb[rus_indices]
            alignments = align_document_faiss(kaz_sents, kaz_emb, rus_sents, rus_emb, config)
            kaz_rus_alignments = {kaz_idx: (rus_idx, sim) for kaz_idx, rus_idx, sim in alignments}
            alignment_stats["aligned_kaz_rus"] += len(alignments)

        # Align with English
        kaz_eng_alignments = {}
        if doc_id in eng_by_doc:
            eng_sents = eng_by_doc[doc_id]
            eng_indices = [eng_idx_map[id(s)] for s in eng_sents]
            eng_emb = all_eng_emb[eng_indices]
            alignments = align_document_faiss(kaz_sents, kaz_emb, eng_sents, eng_emb, config)
            kaz_eng_alignments = {kaz_idx: (eng_idx, sim) for kaz_idx, eng_idx, sim in alignments}
            alignment_stats["aligned_kaz_eng"] += len(alignments)

        # Create triplets
        triplet_idx = 0
        for kaz_idx, kaz_sent in enumerate(kaz_sents):
            rus_match = kaz_rus_alignments.get(kaz_idx)
            eng_match = kaz_eng_alignments.get(kaz_idx)

            if rus_match is None and eng_match is None:
                continue

            rus_sent = rus_by_doc[doc_id][rus_match[0]] if rus_match else None
            eng_sent = eng_by_doc[doc_id][eng_match[0]] if eng_match else None
            sim_kaz_rus = rus_match[1] if rus_match else 0.0
            sim_kaz_eng = eng_match[1] if eng_match else 0.0

            metadata_base = {
                "source_doc": doc_id,
                "kaz_idx": kaz_sent.get("sentence_idx"),
                "year": kaz_sent.get("metadata", {}).get("year"),
                "company": kaz_sent.get("metadata", {}).get("company"),
                "doc_type": kaz_sent.get("metadata", {}).get("type"),
            }

            triplet = create_aligned_triplet(
                kaz_sent["text"],
                rus_sent["text"] if rus_sent else None,
                eng_sent["text"] if eng_sent else None,
                sim_kaz_rus, sim_kaz_eng,
                "1:1", "1:1",
                doc_id, triplet_idx,
                metadata_base,
            )
            all_triplets.append(triplet)
            triplet_idx += 1

            if rus_sent and eng_sent:
                alignment_stats["full_triplets"] += 1
            else:
                alignment_stats["partial_triplets"] += 1

    return all_triplets, alignment_stats


def main():
    parser = argparse.ArgumentParser(
        description="Embed and align parallel sentences using Bertalign + MEXMA"
    )
    parser.add_argument(
        "--input_dir", type=str, default="chunked_sentences",
        help="Input directory with sentence JSONL files"
    )
    parser.add_argument(
        "--output_file", type=str, default="aligned_corpus.jsonl",
        help="Output file for aligned corpus"
    )
    parser.add_argument(
        "--model", type=str, choices=["mexma", "sonar", "labse", "auto"],
        default="mexma",
        help="Embedding model (mexma=SOTA Bertalign, sonar=FAISS 1:1 fallback, labse=Bertalign fallback)"
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Minimum similarity score for alignment"
    )
    parser.add_argument(
        "--min_words", type=int, default=DEFAULT_MIN_WORDS,
        help="Minimum words per sentence"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--max_align", type=int, default=5,
        help="Max sentences in one alignment bead (Bertalign)"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Bertalign top-k parameter for first-pass candidate search"
    )
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="Disable GPU usage"
    )
    parser.add_argument(
        "--compare_only", action="store_true",
        help="Only compare MEXMA vs SONAR vs LaBSE, don't perform alignment"
    )
    args = parser.parse_args()

    # Setup
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / args.input_dir
    output_file = base_dir / args.output_file

    device = "cuda" if not args.no_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = AlignmentConfig(
        similarity_threshold=args.similarity_threshold,
        min_words=args.min_words,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        max_align=args.max_align,
        top_k=args.top_k,
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

    # Compare models if requested
    comparison_results = None
    if args.compare_only or args.model == "auto":
        comparison_results = compare_models(
            kaz_sentences,
            rus_sentences if rus_sentences else eng_sentences,
            sample_size=100,
            device=device
        )

        if args.compare_only:
            results_file = base_dir / "model_comparison_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"\nComparison results saved to: {results_file}")
            return

        args.model = comparison_results["winner"]

    # Run appropriate pipeline
    print(f"\nUsing model: {args.model.upper()}")

    if args.model == "sonar":
        # SONAR uses FAISS 1:1 (not compatible with Bertalign encoder interface)
        if not SONAR_AVAILABLE:
            print("[ERROR] SONAR not available. Install with: pip install sonar-space")
            print("Falling back to MEXMA + Bertalign...")
            args.model = "mexma"
            all_triplets, alignment_stats = run_bertalign_pipeline(
                args, config, device, kaz_sentences, rus_sentences, eng_sentences, base_dir
            )
        else:
            all_triplets, alignment_stats = run_sonar_pipeline(
                args, config, device, kaz_sentences, rus_sentences, eng_sentences, base_dir
            )
    elif args.model in ("mexma", "labse"):
        if not BERTALIGN_AVAILABLE:
            print("[ERROR] bertalign not available. Install with: pip install bertalign")
            return
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[ERROR] sentence-transformers not available.")
            return
        all_triplets, alignment_stats = run_bertalign_pipeline(
            args, config, device, kaz_sentences, rus_sentences, eng_sentences, base_dir
        )
    else:
        print(f"[ERROR] Unknown model: {args.model}")
        return

    # Save aligned corpus
    print(f"\nSaving {len(all_triplets):,} aligned triplets...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in all_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    # Save alignment metadata
    metadata_file = output_file.with_suffix('.meta.json')
    model_details = {
        "mexma": "facebook/MEXMA via Bertalign m:n DP",
        "sonar": "SONAR text_sonar_basic_encoder via FAISS 1:1",
        "labse": "LaBSE via Bertalign m:n DP",
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "model_details": model_details.get(args.model, args.model),
            "config": {
                "similarity_threshold": config.similarity_threshold,
                "min_words": config.min_words,
                "length_ratio_min": config.length_ratio_min,
                "length_ratio_max": config.length_ratio_max,
                "max_align": config.max_align,
                "top_k": config.top_k,
            },
            "stats": alignment_stats,
            "model_comparison": comparison_results
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"  Model:                {args.model.upper()}")
    print(f"  Alignment method:     {'Bertalign m:n DP' if args.model != 'sonar' else 'FAISS 1:1'}")
    print(f"  Total KAZ sentences:  {alignment_stats['total_kaz_sentences']:,}")
    print(f"  Aligned KAZ-RUS:      {alignment_stats['aligned_kaz_rus']:,}")
    print(f"  Aligned KAZ-ENG:      {alignment_stats['aligned_kaz_eng']:,}")
    print(f"  Full triplets:        {alignment_stats['full_triplets']:,}")
    print(f"  Partial triplets:     {alignment_stats['partial_triplets']:,}")
    print(f"  Total aligned:        {len(all_triplets):,}")
    if args.model != "sonar":
        print(f"  Many-to-many KAZ-RUS: {alignment_stats['many_to_many_kaz_rus']:,}")
        print(f"  Many-to-many KAZ-ENG: {alignment_stats['many_to_many_kaz_eng']:,}")
    print(f"\n  Output file: {output_file.absolute()}")
    print(f"  Metadata:    {metadata_file.absolute()}")
    print("\n" + "=" * 60)
    print("Next step: Run post-processing:")
    print("  python scripts/03b_postprocess_v2.py \\")
    print(f"      --input_file {args.output_file} \\")
    print("      --output_file aligned_corpus_clean.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
