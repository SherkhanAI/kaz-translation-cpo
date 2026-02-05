#!/usr/bin/env python3
"""
Phase 6: Evaluation
===================

Evaluate trained translation model using BLEU, chrF, and COMET metrics.

Usage:
    python scripts/06_evaluate.py \
        --model models/qwen-kaz-translator-cpo/final \
        --test_file aligned_corpus.jsonl \
        --output_dir evaluation_results

Author: Dataset for Translator Project
Python: 3.11
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
from comet import download_model, load_from_checkpoint


# Prompt templates (same as training)
PROMPT_TEMPLATES = {
    "kaz2rus": """Translate the following Kazakh text to Russian. Provide only the translation without explanations.

Kazakh: {source}

Russian:""",

    "kaz2eng": """Translate the following Kazakh text to English. Provide only the translation without explanations.

Kazakh: {source}

English:"""
}


def load_model(model_path: str, use_4bit: bool = False):
    """Load model for inference."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


def load_test_data(file_path: str, max_samples: int = None) -> List[dict]:
    """Load test data from aligned corpus."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def generate_translation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256
) -> str:
    """Generate translation."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for evaluation
            pad_token_id=tokenizer.pad_token_id
        )

    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Clean up
    translation = translation.strip()
    for stop_word in ["\n\n", "Kazakh:", "Russian:", "English:"]:
        if stop_word in translation:
            translation = translation.split(stop_word)[0].strip()

    return translation


def compute_bleu(hypotheses: List[str], references: List[str]) -> dict:
    """Compute BLEU score."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "score": bleu.score,
        "precisions": bleu.precisions,
        "bp": bleu.bp,
        "ratio": bleu.ratio
    }


def compute_chrf(hypotheses: List[str], references: List[str]) -> dict:
    """Compute chrF score."""
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {
        "score": chrf.score
    }


def compute_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_path: str = "Unbabel/wmt22-comet-da"
) -> dict:
    """Compute COMET score."""
    try:
        comet_model = load_from_checkpoint(download_model(model_path))

        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]

        output = comet_model.predict(data, batch_size=8, gpus=1)

        return {
            "score": output.system_score,
            "scores": output.scores[:10]  # First 10 sentence-level scores
        }
    except Exception as e:
        print(f"COMET evaluation failed: {e}")
        return {"score": None, "error": str(e)}


def evaluate_direction(
    model,
    tokenizer,
    test_data: List[dict],
    direction: str,
    use_comet: bool = True
) -> dict:
    """Evaluate model on a specific translation direction."""
    source_lang, target_lang = direction.split("2")

    sources = []
    references = []
    hypotheses = []

    for item in tqdm(test_data, desc=f"Evaluating {direction}"):
        source_text = item.get(source_lang)
        reference_text = item.get(target_lang)

        if not source_text or not reference_text:
            continue

        # Generate translation
        prompt = PROMPT_TEMPLATES[direction].format(source=source_text)
        hypothesis = generate_translation(model, tokenizer, prompt)

        sources.append(source_text)
        references.append(reference_text)
        hypotheses.append(hypothesis)

    if len(hypotheses) == 0:
        return {"error": "No valid samples found"}

    # Compute metrics
    results = {
        "direction": direction,
        "num_samples": len(hypotheses),
        "bleu": compute_bleu(hypotheses, references),
        "chrf": compute_chrf(hypotheses, references)
    }

    if use_comet:
        results["comet"] = compute_comet(sources, hypotheses, references)

    # Sample translations for manual inspection
    results["samples"] = []
    for i in range(min(5, len(hypotheses))):
        results["samples"].append({
            "source": sources[i],
            "reference": references[i],
            "hypothesis": hypotheses[i]
        })

    return results


def compare_models(
    model_paths: List[str],
    test_data: List[dict],
    direction: str,
    use_4bit: bool = False
) -> dict:
    """Compare multiple models on the same test data."""
    comparison = {
        "direction": direction,
        "models": {}
    }

    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\n--- Evaluating: {model_name} ---")

        model, tokenizer = load_model(model_path, use_4bit)
        results = evaluate_direction(model, tokenizer, test_data, direction, use_comet=False)

        comparison["models"][model_name] = {
            "bleu": results["bleu"]["score"],
            "chrf": results["chrf"]["score"]
        }

        # Free memory
        del model
        torch.cuda.empty_cache()

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="Path to baseline model for comparison"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="aligned_corpus.jsonl",
        help="Test data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["kaz2rus", "kaz2eng"],
        help="Translation directions to evaluate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--no_comet",
        action="store_true",
        help="Skip COMET evaluation"
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_file = base_dir / args.test_file
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Load test data
    print(f"Loading test data from: {test_file}")
    test_data = load_test_data(str(test_file), args.max_samples)
    print(f"Loaded {len(test_data):,} test samples")

    # Load model
    model, tokenizer = load_model(args.model, args.use_4bit)

    # Evaluate
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "test_file": str(test_file),
        "num_samples": len(test_data),
        "directions": {}
    }

    for direction in args.directions:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {direction}")
        print("=" * 60)

        results = evaluate_direction(
            model, tokenizer, test_data, direction,
            use_comet=not args.no_comet
        )

        all_results["directions"][direction] = results

        # Print summary
        print(f"\n  BLEU:  {results['bleu']['score']:.2f}")
        print(f"  chrF:  {results['chrf']['score']:.2f}")
        if "comet" in results and results["comet"]["score"]:
            print(f"  COMET: {results['comet']['score']:.4f}")

    # Compare with baseline if provided
    if args.baseline_model:
        print(f"\n{'=' * 60}")
        print("Comparing with baseline...")
        print("=" * 60)

        # Free memory from main model
        del model
        torch.cuda.empty_cache()

        for direction in args.directions:
            comparison = compare_models(
                [args.baseline_model, args.model],
                test_data,
                direction,
                args.use_4bit
            )
            all_results[f"comparison_{direction}"] = comparison

            # Print comparison
            print(f"\n{direction}:")
            for model_name, scores in comparison["models"].items():
                print(f"  {model_name}: BLEU={scores['bleu']:.2f}, chrF={scores['chrf']:.2f}")

    # Save results
    output_file = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_file.absolute()}")

    # Print final summary
    print("\nSUMMARY:")
    for direction, results in all_results["directions"].items():
        print(f"\n  {direction}:")
        print(f"    BLEU:  {results['bleu']['score']:.2f}")
        print(f"    chrF:  {results['chrf']['score']:.2f}")


if __name__ == "__main__":
    main()
