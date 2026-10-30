import json
import os
import re

import requests
from flask import current_app

import app

MODEL = "gpt-4o"
LABELS = ("Subject", "Predicate", "Object")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "annotation_prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPT = f.read()


def parse_records(text):
    """Turn the model reply into a list of token records (dicts)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text if text.startswith("[") else f"[{text}]")
        return [r for r in data if isinstance(r, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: keep every record that parses on its own, skip malformed ones
    records = []
    for chunk in re.findall(r"\{[^{}]*\}", text):
        try:
            records.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return records


def records_to_triples(records):
    """Group token records by sentence_id and join tokens per label into triples."""
    groups = {}
    for record in records:
        groups.setdefault(record.get("sentence_id"), []).append(record)

    triples = []
    for group in groups.values():
        parts = {label: [] for label in LABELS}
        for record in group:
            label = record.get("spo_label")
            if label in parts:
                parts[label].append(str(record.get("token", "")))
        if any(parts.values()):
            triples.append({
                "subject": " ".join(parts["Subject"]),
                "predicate": " ".join(parts["Predicate"]),
                "object": " ".join(parts["Object"]),
            })
    return triples


def extract_triples(sentence_data):
    sentence = sentence_data["sentence"]
    try:
        response = app.llm.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Sentence ID: 1\n{sentence}"},
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        current_app.logger.error(f"OpenAI request failed: {e}")
        return {"triples": []}

    current_app.logger.debug(f"raw model output: {content}")
    return {"triples": records_to_triples(parse_records(content))}


def send_triples(sentence_data):
    payload = extract_triples(sentence_data)
    payload['sentence_data'] = sentence_data
    current_app.logger.debug(f"payload: {payload}")
    reasoner_address = current_app.config.get('REASONER_ADDRESS', None)
    if reasoner_address:
        requests.post(f"http://{reasoner_address}/process", json=payload)
