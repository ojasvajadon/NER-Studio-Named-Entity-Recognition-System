"""
train_ner.py  —  Improved NER training script with early stopping,
                 synthetic data support, and better default hyperparameters.

Usage (from project root):
    python train_ner.py

With synthetic data (recommended):
    python train_ner.py --synthetic data/synthetic_ner.jsonl

For maximum accuracy (needs en_core_web_lg installed):
    python train_ner.py --base-model en_core_web_lg --synthetic data/synthetic_ner.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Iterable

import spacy
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
from spacy.util import compounding, minibatch

# ── label maps ───────────────────────────────────────────────────────────────

LABEL_MAP = {
    "per": "PERSON",
    "org": "ORG",
    "geo": "GPE",
    "gpe": "GPE",
    "loc": "LOC",   # ← was missing; now location tags survive
    "tim": "DATE",
    "eve": "EVENT",
    "art": "WORK_OF_ART",
    "nat": "NORP",
}
HF_TAG_LABEL_MAP = {
    "per": "PERSON",
    "person": "PERSON",
    "org": "ORG",
    "organization": "ORG",
    "corporation": "ORG",
    "group": "NORP",
    "loc": "GPE",
    "location": "GPE",
    "geo": "GPE",
    "gpe": "GPE",
    "tim": "DATE",
    "date": "DATE",
    "eve": "EVENT",
    "event": "EVENT",
    "art": "WORK_OF_ART",
    "creative-work": "WORK_OF_ART",
    "building": "FAC",
    "facility": "FAC",
    "nat": "NORP",
    "misc": "NORP",
    "other": "NORP",
    "product": "PRODUCT",
}
EXTRA_DATASETS = {
    "conll2003": {"hf_name": "tomaarsen/conll2003", "config": None},
    "hf_conll2003": {"hf_name": "tomaarsen/conll2003", "config": None},
    "wikiann_en": {"hf_name": "wikiann", "config": "en"},
    "fewnerd_supervised": {"hf_name": "DFKI-SLT/few-nerd", "config": "supervised"},
}

NO_SPACE_BEFORE = {".", ",", ":", ";", "!", "?", "%", ")", "]", "}", "'s", "n't", "''"}
NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}


# ── text utilities ────────────────────────────────────────────────────────────

def infer_spaces(words: list[str]) -> list[bool]:
    spaces = [True] * len(words)
    if not words:
        return spaces
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        no_space = next_word in NO_SPACE_BEFORE or next_word.startswith("'") or current_word in NO_SPACE_AFTER
        spaces[i] = not no_space
    spaces[-1] = False
    return spaces


def words_to_text_with_offsets(words: list[str]) -> tuple[str, list[int], list[int]]:
    spaces = infer_spaces(words)
    chunks: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for word, has_space in zip(words, spaces):
        starts.append(cursor)
        chunks.append(word)
        cursor += len(word)
        ends.append(cursor)
        if has_space:
            chunks.append(" ")
            cursor += 1
    text = "".join(chunks)
    return text, starts, ends


# ── label resolution ──────────────────────────────────────────────────────────

def _resolve_label(raw_label: str, label_map: dict[str, str]) -> str | None:
    normalized = raw_label.strip()
    if not normalized:
        return None
    for key in (normalized, normalized.lower(), normalized.upper()):
        mapped = label_map.get(key)
        if mapped:
            return mapped
    return label_map.get(normalized.lower())


# ── IOB parsing ───────────────────────────────────────────────────────────────

def parse_iob_entities(tags: list[str], label_map: dict[str, str]) -> list[tuple[int, int, str]]:
    entities: list[tuple[int, int, str]] = []
    current_start = None
    current_label = None
    previous_plain_label: str | None = None

    for idx, raw_tag in enumerate(tags):
        tag = (raw_tag or "O").strip()
        if not tag or tag.upper() == "O":
            if current_start is not None and current_label is not None:
                entities.append((current_start, idx, current_label))
                current_start = None
                current_label = None
            previous_plain_label = None
            continue

        if "-" in tag:
            prefix, raw_label = tag.split("-", 1)
        else:
            raw_label = tag
            prefix = "I" if previous_plain_label == raw_label else "B"

        mapped_label = _resolve_label(raw_label, label_map)
        previous_plain_label = raw_label
        if mapped_label is None:
            if current_start is not None and current_label is not None:
                entities.append((current_start, idx, current_label))
                current_start = None
                current_label = None
            previous_plain_label = None
            continue

        prefix = prefix.upper()
        if prefix == "B":
            if current_start is not None and current_label is not None:
                entities.append((current_start, idx, current_label))
            current_start = idx
            current_label = mapped_label
        elif prefix == "I":
            if current_start is None or current_label != mapped_label:
                current_start = idx
                current_label = mapped_label
        else:
            if current_start is not None and current_label is not None:
                entities.append((current_start, idx, current_label))
            current_start = idx
            current_label = mapped_label

    if current_start is not None and current_label is not None:
        entities.append((current_start, len(tags), current_label))

    return entities


# ── CSV loader ────────────────────────────────────────────────────────────────

def conll_csv_to_examples(
    dataset_path: Path, max_sentences: int
) -> list[tuple[str, dict[str, list[tuple[int, int, str]]]]]:
    examples: list[tuple[str, dict[str, list[tuple[int, int, str]]]]] = []

    with dataset_path.open("r", encoding="latin-1", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        words: list[str] = []
        tags: list[str] = []

        def flush_sentence() -> None:
            if not words:
                return
            text, starts, ends = words_to_text_with_offsets(words)
            token_spans = parse_iob_entities(tags, LABEL_MAP)
            entities = []
            for start_i, end_i, label in token_spans:
                char_start = starts[start_i]
                char_end = ends[end_i - 1]
                if char_start < char_end:
                    entities.append((char_start, char_end, label))
            examples.append((text, {"entities": entities}))
            words.clear()
            tags.clear()

        for row in reader:
            sentence_marker = (row.get("Sentence #") or "").strip()
            if sentence_marker:
                flush_sentence()
                if len(examples) >= max_sentences:
                    break
            word = (row.get("Word") or "").strip()
            tag = (row.get("Tag") or "O").strip()
            if not word:
                continue
            words.append(word)
            tags.append(tag)

        if len(examples) < max_sentences:
            flush_sentence()

    return examples


# ── HuggingFace loader ────────────────────────────────────────────────────────

def _safe_load_dataset(name: str, config: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for --extra-datasets. "
            "Install it with: pip install datasets"
        ) from exc
    return load_dataset(name, config)


def hf_dataset_to_examples(
    dataset_key: str, max_per_split: int, seed: int
) -> list[tuple[str, dict[str, list[tuple[int, int, str]]]]]:
    config = EXTRA_DATASETS.get(dataset_key)
    if config is None:
        raise ValueError(
            f"Unknown extra dataset '{dataset_key}'. Supported: {', '.join(sorted(EXTRA_DATASETS))}"
        )
    ds = _safe_load_dataset(config["hf_name"], config["config"])
    examples: list[tuple[str, dict[str, list[tuple[int, int, str]]]]] = []
    split_counts: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        split_data = ds[split]
        if len(split_data) == 0:
            continue
        take_n = min(max_per_split, len(split_data))
        split_data = split_data.shuffle(seed=seed).select(range(take_n))

        tag_feature = split_data.features["ner_tags"]
        tag_names = getattr(getattr(tag_feature, "feature", None), "names", None)
        if not tag_names:
            raise ValueError(
                f"Unsupported ner_tags format for dataset '{dataset_key}' split '{split}'"
            )

        split_added = 0
        for row in split_data:
            words = [w for w in row.get("tokens", []) if isinstance(w, str)]
            tag_ids = row.get("ner_tags", [])
            if not words or len(words) != len(tag_ids):
                continue
            tags: list[str] = []
            for tag_id in tag_ids:
                if not isinstance(tag_id, int) or tag_id < 0 or tag_id >= len(tag_names):
                    tags.append("O")
                else:
                    tags.append(tag_names[tag_id])
            text, starts, ends = words_to_text_with_offsets(words)
            token_spans = parse_iob_entities(tags, HF_TAG_LABEL_MAP)
            entities = []
            for start_i, end_i, label in token_spans:
                char_start = starts[start_i]
                char_end = ends[end_i - 1]
                if char_start < char_end:
                    entities.append((char_start, char_end, label))
            if not text.strip():
                continue
            examples.append((text, {"entities": entities}))
            split_added += 1
        split_counts[split] = split_added

    stats = ", ".join(f"{k}:{v}" for k, v in split_counts.items()) or "no splits used"
    print(f"Loaded extra dataset '{dataset_key}' -> {stats}")
    return examples


# ── feedback / synthetic loaders ──────────────────────────────────────────────

def load_feedback_examples(
    feedback_path: Path,
) -> list[tuple[str, dict[str, list[tuple[int, int, str]]]]]:
    if not feedback_path.exists():
        return []
    examples: list[tuple[str, dict[str, list[tuple[int, int, str]]]]] = []
    with feedback_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (item.get("text") or "").strip()
            raw_entities = item.get("entities") or []
            if not text or not isinstance(raw_entities, list):
                continue
            entities: list[tuple[int, int, str]] = []
            for ent in raw_entities:
                if not isinstance(ent, list) or len(ent) != 3:
                    continue
                start, end, label = ent
                if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
                    continue
                if 0 <= start < end <= len(text):
                    entities.append((start, end, label))
            examples.append((text, {"entities": entities}))
    return examples


# ── deduplication & sanitation ────────────────────────────────────────────────

def deduplicate_examples(
    examples: list[tuple[str, dict[str, list[tuple[int, int, str]]]]],
) -> list[tuple[str, dict[str, list[tuple[int, int, str]]]]]:
    seen: set[tuple[str, tuple[tuple[int, int, str], ...]]] = set()
    unique: list[tuple[str, dict[str, list[tuple[int, int, str]]]]] = []
    for text, ann in examples:
        entities = sorted((int(s), int(e), str(l)) for s, e, l in ann.get("entities", []))
        key = (text, tuple(entities))
        if key in seen:
            continue
        seen.add(key)
        unique.append((text, {"entities": entities}))
    return unique


def sanitize_examples_for_tokenizer(
    nlp: spacy.Language,
    examples: list[tuple[str, dict[str, list[tuple[int, int, str]]]]],
) -> tuple[list[tuple[str, dict[str, list[tuple[int, int, str]]]]], int]:
    sanitized: list[tuple[str, dict[str, list[tuple[int, int, str]]]]] = []
    dropped_entities = 0

    for text, ann in examples:
        raw_entities = ann.get("entities", [])
        if not raw_entities:
            sanitized.append((text, {"entities": []}))
            continue
        doc = nlp.make_doc(text)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"spacy\.training\.iob_utils",
                message=r".*\[W030\].*",
            )
            biluo = offsets_to_biluo_tags(doc, raw_entities)
        if "-" in biluo:
            valid_entities: list[tuple[int, int, str]] = []
            for start, end, label in raw_entities:
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is None:
                    dropped_entities += 1
                    continue
                valid_entities.append((span.start_char, span.end_char, label))
            sanitized.append((text, {"entities": valid_entities}))
        else:
            sanitized.append((text, {"entities": list(raw_entities)}))

    return sanitized, dropped_entities


# ── train / dev split ─────────────────────────────────────────────────────────

def split_train_dev(data: list, dev_ratio: float, seed: int) -> tuple[list, list]:
    random.Random(seed).shuffle(data)
    dev_size = max(1, int(len(data) * dev_ratio))
    dev = data[:dev_size]
    train = data[dev_size:]
    if not train:
        train, dev = dev, []
    return train, dev


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_ner(
    nlp: spacy.Language,
    dev_data: Iterable[tuple[str, dict]],
    max_samples: int = 4000,
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) on dev set."""
    dev_list = list(dev_data)[:max_samples]
    if not dev_list:
        return 0.0, 0.0, 0.0

    tp = fp = fn = 0
    for text, ann in dev_list:
        pred_doc = nlp(text)
        pred_set = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}
        gold_set = set(ann.get("entities", []))
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ── main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ─ load data ─
    dataset_examples = conll_csv_to_examples(args.dataset, args.max_sentences)
    feedback_examples = load_feedback_examples(args.feedback)
    synthetic_examples = load_feedback_examples(args.synthetic) if args.synthetic else []
    extra_examples: list[tuple[str, dict[str, list[tuple[int, int, str]]]]] = []

    for dataset_key in args.extra_datasets:
        extra_examples.extend(
            hf_dataset_to_examples(dataset_key, args.extra_max_per_split, args.seed)
        )

    feedback_weight = max(1, args.feedback_boost)
    all_examples = (
        dataset_examples
        + extra_examples
        + (feedback_examples * feedback_weight)
        + (synthetic_examples * max(1, args.synthetic_boost))
    )
    all_examples = [item for item in all_examples if item[0].strip()]
    all_examples = deduplicate_examples(all_examples)

    if len(all_examples) < 5:
        raise ValueError("Not enough training data. Need at least 5 examples.")

    print(
        "Training pool sizes -> "
        f"base_csv={len(dataset_examples)}, extra={len(extra_examples)}, "
        f"feedback={len(feedback_examples)} (x{feedback_weight}), "
        f"synthetic={len(synthetic_examples)} (x{max(1, args.synthetic_boost)}), "
        f"merged_unique={len(all_examples)}"
    )

    train_data, dev_data = split_train_dev(all_examples, args.dev_ratio, args.seed)
    print(f"Train: {len(train_data)} | Dev: {len(dev_data)}")

    # ─ load base model ─
    use_blank = args.base_model.strip().lower() in {"blank", "blank_en", "spacy.blank"}
    loaded_pretrained = False
    if use_blank:
        nlp = spacy.blank("en")
        print("Using blank English model as requested.")
    else:
        try:
            nlp = spacy.load(args.base_model)
            loaded_pretrained = True
            print(f"Loaded base model: {args.base_model}")
        except OSError:
            nlp = spacy.blank("en")
            print(f"Base model '{args.base_model}' not found; falling back to blank English model.")

    # ─ set up NER pipe ─
    if "ner" in nlp.pipe_names:
        ner = nlp.get_pipe("ner")
    else:
        ner = nlp.add_pipe("ner")

    all_examples, dropped_entities = sanitize_examples_for_tokenizer(nlp, all_examples)
    if dropped_entities:
        print(f"Dropped {dropped_entities} misaligned entity spans during sanitization.")

    labels: set[str] = set()
    for _, ann in all_examples:
        for _, _, label in ann.get("entities", []):
            labels.add(label)
    for label in sorted(labels):
        ner.add_label(label)
    print(f"NER labels ({len(labels)}): {', '.join(sorted(labels))}")

    # ─ training ─
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    best_f1 = 0.0
    no_improve_count = 0
    best_model_path = args.output.parent / (args.output.name + "_best")

    with nlp.disable_pipes(*other_pipes):
        if loaded_pretrained:
            optimizer = nlp.resume_training()
        else:
            optimizer = nlp.initialize(
                lambda: [
                    Example.from_dict(nlp.make_doc(text), ann)
                    for text, ann in train_data[:200]
                ]
            )

        print(f"\n{'Epoch':>5} | {'Loss':>10} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} |")
        print("-" * 50)

        for i in range(1, args.iterations + 1):
            random.shuffle(train_data)
            losses: dict = {}
            batches = minibatch(train_data, size=compounding(8.0, 64.0, 1.35))
            for batch in batches:
                examples_batch = [
                    Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch
                ]
                nlp.update(examples_batch, sgd=optimizer, drop=args.dropout, losses=losses)

            prec, rec, f1 = evaluate_ner(nlp, dev_data, max_samples=args.eval_samples)
            loss_val = losses.get("ner", 0.0)
            print(f"{i:>5} | {loss_val:>10.4f} | {prec:>7.4f} | {rec:>7.4f} | {f1:>7.4f} |")

            if f1 > best_f1:
                best_f1 = f1
                no_improve_count = 0
                # Save best checkpoint
                if best_model_path.exists():
                    shutil.rmtree(best_model_path)
                best_model_path.mkdir(parents=True, exist_ok=True)
                nlp.to_disk(best_model_path)
            else:
                no_improve_count += 1

            if args.early_stopping > 0 and no_improve_count >= args.early_stopping:
                print(f"\n⚑  Early stopping: no improvement for {args.early_stopping} epochs.")
                break

    print(f"\n✓ Best dev F1: {best_f1:.4f}")

    # Restore best model weights into the final output path
    if best_model_path.exists():
        if args.output.exists():
            shutil.rmtree(args.output)
        shutil.copytree(best_model_path, args.output)
        shutil.rmtree(best_model_path)
        print(f"✓ Saved best model to: {args.output}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(args.output)
        print(f"✓ Saved model to: {args.output}")


# ── argument parser ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom spaCy NER model with early stopping and synthetic data support."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("templates/ner_dataset.csv"),
        help="Path to CoNLL-style CSV",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=Path("data/ner_feedback.jsonl"),
        help="Path to saved UI corrections",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=None,
        help="Path to synthetic data JSONL (e.g. data/synthetic_ner.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ner_custom"),
        help="Output model directory",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=50000,
        help="Max sentences loaded from dataset CSV",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
        help="Dropout rate during training",
    )
    parser.add_argument(
        "--feedback-boost",
        type=int,
        default=5,
        help="How many times to repeat feedback examples",
    )
    parser.add_argument(
        "--synthetic-boost",
        type=int,
        default=2,
        help="How many times to repeat synthetic examples",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=4000,
        help="Max dev samples for evaluation per epoch",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=4,
        help="Stop if dev F1 does not improve for this many epochs (0 = disabled)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--extra-datasets",
        nargs="*",
        default=[],
        help=(
            "Extra HuggingFace datasets to download and merge. "
            "Supported: conll2003, wikiann_en, fewnerd_supervised. "
            "Leave empty to skip (saves bandwidth)."
        ),
    )
    parser.add_argument(
        "--extra-max-per-split",
        type=int,
        default=12000,
        help="Max sentences from each split of each extra dataset",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="en_core_web_sm",
        help="Base spaCy model to fine-tune (use en_core_web_lg for better vectors)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
