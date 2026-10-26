import app.util as util


def test_extract_triples_returns_empty_placeholder(sample_sentence_data):
    result = util.extract_triples(sample_sentence_data)
    assert result == {"triples": []}
