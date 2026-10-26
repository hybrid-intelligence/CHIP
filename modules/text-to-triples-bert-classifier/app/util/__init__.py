from flask import current_app
import requests


def extract_triples(sentence_data):
    # TODO: implement actual triple extraction logic
    return {"triples": []}


def send_triples(sentence_data):
    payload = extract_triples(sentence_data)
    payload['sentence_data'] = sentence_data
    current_app.logger.debug(f"payload: {payload}")
    reasoner_address = current_app.config.get('REASONER_ADDRESS', None)
    if reasoner_address:
        requests.post(f"http://{reasoner_address}/process", json=payload)
