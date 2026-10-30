from unittest.mock import Mock

import app
import app.util as util


def make_record(sentence_id, token_id, token, label):
    return {
        "sentence_id": sentence_id,
        "token_id": token_id,
        "sentence": "I take insulin.",
        "token": token,
        "spo_label": label,
    }


RECORDS = [
    make_record(1, 1, "I", "Subject"),
    make_record(1, 2, "take", "Predicate"),
    make_record(1, 3, "insulin", "Object"),
    make_record(1, 4, ".", "other"),
]


def test_records_to_triples_joins_tokens_per_label():
    assert util.records_to_triples(RECORDS) == [
        {"subject": "I", "predicate": "take", "object": "insulin"}
    ]


def test_records_to_triples_drops_groups_with_only_other():
    records = [make_record(1, 1, "Thanks", "other"), make_record(1, 2, ".", "other")]
    assert util.records_to_triples(records) == []


def test_records_to_triples_separates_sentence_ids():
    records = RECORDS + [
        make_record(2, 1, "I", "Subject"),
        make_record(2, 2, "eat", "Predicate"),
        make_record(2, 3, "oatmeal", "Object"),
    ]
    triples = util.records_to_triples(records)
    assert len(triples) == 2
    assert triples[1]["predicate"] == "eat"


def test_parse_records_handles_code_fence_and_missing_brackets():
    text = '```json\n{"sentence_id": 1, "token": "I", "spo_label": "Subject"},\n{"sentence_id": 1, "token": "take", "spo_label": "Predicate"}\n```'
    records = util.parse_records(text)
    assert [r["token"] for r in records] == ["I", "take"]


def test_parse_records_skips_malformed_records():
    text = '[{"sentence_id": 1, "token": "I", "spo_label": "Subject"}, {"token": "":"", "spo_label": "other"}]'
    records = util.parse_records(text)
    assert len(records) == 1
    assert records[0]["token"] == "I"


def test_extract_triples_with_mocked_client(application, sample_sentence_data, monkeypatch):
    import json

    reply = Mock()
    reply.choices = [Mock(message=Mock(content=json.dumps(RECORDS)))]
    client = Mock()
    client.chat.completions.create.return_value = reply
    monkeypatch.setattr(app.llm, "client", client)

    with application.app_context():
        result = util.extract_triples(sample_sentence_data)

    assert result == {"triples": [{"subject": "I", "predicate": "take", "object": "insulin"}]}


def test_extract_triples_returns_empty_when_api_fails(application, sample_sentence_data, monkeypatch):
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    monkeypatch.setattr(app.llm, "client", client)

    with application.app_context():
        result = util.extract_triples(sample_sentence_data)

    assert result == {"triples": []}
