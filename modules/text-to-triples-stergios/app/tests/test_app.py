from unittest.mock import patch


def test_hello(client):
    response = client.get('/')
    assert b'Hello' in response.data


def test_new_sentence(client, sample_sentence_data):
    with patch('app.util.send_triples') as st:
        res = client.post('/process', json=sample_sentence_data)
        st.assert_called_once()
        assert res.status_code == 200
