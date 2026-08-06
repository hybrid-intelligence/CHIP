import pytest
from app import create_app
from unittest.mock import Mock


@pytest.fixture()
def reasoner_address():
    return "dummy"


@pytest.fixture()
def application(monkeypatch, reasoner_address):
    monkeypatch.setenv("REASONER_MODULE", "TEST_MOD_1")
    monkeypatch.setenv("TEST_MOD_1", reasoner_address)
    yield create_app(test=True)


@pytest.fixture()
def client(application):
    tc = application.test_client()
    # For detecting errors and disabling logging in general
    setattr(tc.application, "logger", Mock(tc.application.logger))
    return tc


@pytest.fixture()
def sample_sentence():
    return "Some cool sentence."


@pytest.fixture()
def sample_name():
    return 'John'


@pytest.fixture()
def sample_timestamp():
    return '...'


@pytest.fixture()
def sample_sentence_data(sample_sentence, sample_name, sample_timestamp):
    return {
        'patient_name': sample_name,
        'sentence': sample_sentence,
        'timestamp': sample_timestamp
    }
