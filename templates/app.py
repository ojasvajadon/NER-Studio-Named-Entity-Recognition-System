from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import spacy
from flask import Flask, render_template, request
from spacy import displacy
from spacy.util import filter_spans

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "models" / "ner_custom"
DATA_DIR = PROJECT_DIR / "data"
FEEDBACK_FILE = DATA_DIR / "ner_feedback.jsonl"

LABEL_OPTIONS = [
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "FAC",
    "EVENT",
    "PRODUCT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
]

GENERIC_TECH_TERMS = {"AI", "ML", "NLP", "DL", "LLM", "LLMS", "GENAI"}
BACKFILL_LABELS = {"MONEY", "PERCENT", "TIME", "QUANTITY", "CARDINAL", "ORDINAL"}
KNOWN_GPE_TERMS = {
    "Europe",
    "Asia",
    "Africa",
    "North America",
    "South America",
    "Middle East",
    "India",
    "China",
    "Japan",
    "United States",
    "United Kingdom",
}
PERSON_BLOCKLIST = {
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
ORG_HINT_WORDS = {
    "Inc",
    "Ltd",
    "LLC",
    "Corp",
    "Corporation",
    "Company",
    "Technologies",
    "Technology",
    "University",
    "Institute",
    "Bank",
    "Agency",
    "Committee",
    "Ministry",
}
NAME_CONTEXT_PATTERN = re.compile(
    r"(?i:\b(?:my name is|i am|i'm|this is)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
)
LOCATION_CONTEXT_PATTERN = re.compile(
    r"(?i:\b(?:live|lives|lived|stay|stays|stayed|reside|resides|resided|work|works|worked|based|born)\s+(?:in|at|from)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
)

app = Flask(__name__, template_folder=str(BASE_DIR))
backup_nlp = None


if MODEL_DIR.exists():
    nlp = spacy.load(str(MODEL_DIR))
    model_status = f"custom ({MODEL_DIR})"
    try:
        backup_nlp = spacy.load("en_core_web_sm")
    except OSError:
        backup_nlp = None
else:
    try:
        nlp = spacy.load("en_core_web_sm")
        model_status = "en_core_web_sm"
    except OSError:
        nlp = spacy.blank("en")
        model_status = "blank_en"


def _is_probable_person(phrase: str) -> bool:
    words = [w for w in phrase.replace(".", "").split() if w]
    if not 2 <= len(words) <= 3:
        return False
    if any(w in PERSON_BLOCKLIST for w in words):
        return False
    if any(w.upper() in GENERIC_TECH_TERMS for w in words):
        return False
    if any(w in ORG_HINT_WORDS for w in words):
        return False
    return all(re.match(r"^[A-Z][a-z]+$", w) for w in words)


def _is_title_phrase(phrase: str) -> bool:
    words = [w for w in phrase.split() if w]
    if not words:
        return False
    return all(re.fullmatch(r"[A-Z][a-z]+", w) for w in words)


def _trim_stray_i_tokens(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    changed = True
    while changed and start < end:
        changed = False
        entity_text = text[start:end]

        if entity_text.startswith("I ") and end - (start + 2) > 1:
            start += 2
            changed = True
        elif entity_text.endswith(" I") and (end - 2) - start > 1:
            end -= 2
            changed = True

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

    return start, end


def _context_entities(text: str) -> list[dict[str, Any]]:
    suggested: list[dict[str, Any]] = []

    for match in NAME_CONTEXT_PATTERN.finditer(text):
        start, end = match.span(1)
        candidate = text[start:end]
        if not _is_title_phrase(candidate):
            continue
        suggested.append(
            {
                "text": candidate,
                "label": "PERSON",
                "start": start,
                "end": end,
                "source": "rule",
            }
        )

    for match in LOCATION_CONTEXT_PATTERN.finditer(text):
        start, end = match.span(1)
        candidate = text[start:end]
        if not _is_title_phrase(candidate):
            continue
        if any(word in ORG_HINT_WORDS for word in candidate.split()):
            continue
        suggested.append(
            {
                "text": candidate,
                "label": "GPE",
                "start": start,
                "end": end,
                "source": "rule",
            }
        )

    return suggested


def _overlaps(start: int, end: int, entities: list[dict[str, Any]]) -> bool:
    return any(start < ent["end"] and end > ent["start"] for ent in entities)


def _apply_accuracy_rules(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrected: list[dict[str, Any]] = []
    context_rules = _context_entities(text)
    context_by_span = {(ent["start"], ent["end"]): ent["label"] for ent in context_rules}

    for ent in sorted(entities, key=lambda item: (item["start"], item["end"])):
        ent = ent.copy()
        if ent["label"] in {"PERSON", "ORG", "LOC", "GPE"}:
            new_start, new_end = _trim_stray_i_tokens(text, ent["start"], ent["end"])
            if new_start < new_end and (new_start != ent["start"] or new_end != ent["end"]):
                ent["start"] = new_start
                ent["end"] = new_end
                ent["text"] = text[new_start:new_end]
                ent["source"] = "rule"

        entity_text = ent["text"].strip()
        clean_upper = re.sub(r"[^A-Za-z]", "", entity_text).upper()

        if clean_upper in GENERIC_TECH_TERMS:
            continue

        forced_label = context_by_span.get((ent["start"], ent["end"]))
        if forced_label and ent["label"] != forced_label:
            ent["label"] = forced_label
            ent["source"] = "rule"

        if ent["label"] == "LOC" and entity_text in KNOWN_GPE_TERMS:
            ent["label"] = "GPE"
            ent["source"] = "rule"

        if ent["label"] in {"ORG", "LOC", "GPE"} and _is_probable_person(entity_text):
            ent["label"] = "PERSON"
            ent["source"] = "rule"

        corrected.append(ent)

    for ent in context_rules:
        if _overlaps(ent["start"], ent["end"], corrected):
            continue
        corrected.append(ent)

    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text):
        start, end = match.span()
        candidate = match.group(0)

        if _overlaps(start, end, corrected):
            continue
        if not _is_probable_person(candidate):
            continue

        corrected.append(
            {
                "text": candidate,
                "label": "PERSON",
                "start": start,
                "end": end,
                "source": "rule",
            }
        )

    best_by_span: dict[tuple[int, int], dict[str, Any]] = {}
    for ent in corrected:
        key = (ent["start"], ent["end"])
        existing = best_by_span.get(key)
        if existing is None or ent.get("source") == "rule":
            best_by_span[key] = ent

    return sorted(best_by_span.values(), key=lambda item: (item["start"], item["end"]))


def _merge_backup_entities(
    primary_entities: list[dict[str, Any]], backup_entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = sorted(primary_entities, key=lambda item: (item["start"], item["end"]))
    for ent in backup_entities:
        if ent["label"] not in BACKFILL_LABELS:
            continue
        if _overlaps(ent["start"], ent["end"], merged):
            continue
        ent["source"] = "backup"
        merged.append(ent)
    return sorted(merged, key=lambda item: (item["start"], item["end"]))


def _render_entities(text: str, entities: list[dict[str, Any]]) -> str:
    doc = nlp.make_doc(text)
    spans = []
    for ent in entities:
        span = doc.char_span(ent["start"], ent["end"], label=ent["label"], alignment_mode="contract")
        if span is not None:
            spans.append(span)
    doc.ents = filter_spans(spans)
    return displacy.render(doc, style="ent", page=False)


def analyze_text(text: str) -> dict[str, Any]:
    doc = nlp(text)
    model_entities = [
        {
            "id": idx,
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "source": "model",
        }
        for idx, ent in enumerate(doc.ents, start=1)
    ]

    if backup_nlp is not None:
        backup_doc = backup_nlp(text)
        backup_entities = [
            {
                "id": idx,
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "backup_raw",
            }
            for idx, ent in enumerate(backup_doc.ents, start=1)
        ]
        model_entities = _merge_backup_entities(model_entities, backup_entities)

    entities = _apply_accuracy_rules(text, model_entities)
    for idx, ent in enumerate(entities, start=1):
        ent["id"] = idx

    label_counts = dict(sorted(Counter(ent["label"] for ent in entities).items()))
    html = _render_entities(text, entities)

    return {
        "text": text,
        "html": html,
        "entities": entities,
        "label_counts": label_counts,
        "entity_total": len(entities),
    }


def _base_context() -> dict[str, Any]:
    return {
        "model_status": model_status,
        "label_options": LABEL_OPTIONS,
        "text": "",
        "html": None,
        "entities": [],
        "label_counts": {},
        "entity_total": 0,
        "error": None,
        "notice": None,
    }


@app.route("/")
def index() -> str:
    return render_template("index.html", **_base_context())


@app.route("/entity", methods=["POST"])
def entity() -> str:
    typed_text = (request.form.get("text") or "").strip()
    uploaded_file = request.files.get("file")

    file_text = ""
    if uploaded_file and uploaded_file.filename:
        file_text = uploaded_file.read().decode("utf-8", errors="ignore").strip()

    text = typed_text or file_text
    if not text:
        context = _base_context()
        context["error"] = "Please paste text or upload a .txt file before running analysis."
        return render_template("index.html", **context)

    context = _base_context()
    context.update(analyze_text(text))
    return render_template("index.html", **context)


@app.route("/save-feedback", methods=["POST"])
def save_feedback() -> str:
    text = (request.form.get("text") or "").strip()
    starts = request.form.getlist("entity_start")
    ends = request.form.getlist("entity_end")
    labels = request.form.getlist("entity_label")

    if not text:
        context = _base_context()
        context["error"] = "No text found for feedback. Please run analysis again."
        return render_template("index.html", **context)

    corrected_entities: list[dict[str, Any]] = []
    for start_raw, end_raw, label in zip(starts, ends, labels):
        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError:
            continue

        if start < 0 or end <= start or end > len(text):
            continue
        if label not in LABEL_OPTIONS:
            continue

        corrected_entities.append(
            {
                "text": text[start:end],
                "label": label,
                "start": start,
                "end": end,
                "source": "human",
            }
        )

    if not corrected_entities:
        context = _base_context()
        context.update(analyze_text(text))
        context["error"] = "No valid entity corrections were submitted."
        return render_template("index.html", **context)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "entities": [[ent["start"], ent["end"], ent["label"]] for ent in corrected_entities],
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

    context = _base_context()
    corrected_entities = sorted(corrected_entities, key=lambda item: (item["start"], item["end"]))
    for idx, ent in enumerate(corrected_entities, start=1):
        ent["id"] = idx

    context.update(
        {
            "text": text,
            "entities": corrected_entities,
            "entity_total": len(corrected_entities),
            "label_counts": dict(sorted(Counter(ent["label"] for ent in corrected_entities).items())),
            "html": _render_entities(text, corrected_entities),
            "notice": f"Saved {len(corrected_entities)} corrected entities to {FEEDBACK_FILE}.",
        }
    )
    return render_template("index.html", **context)


if __name__ == "__main__":
    # use_reloader=False prevents the Flask reloader from spawning a child
    # process that tries to register signal handlers outside the main thread
    # (crashes with Python 3.11+ inside certain virtual-environment setups).
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
