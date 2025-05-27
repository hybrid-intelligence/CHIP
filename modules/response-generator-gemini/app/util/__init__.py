from flask import current_app
from enum import auto
from strenum import StrEnum
from app.util.gemini import generate
import requests

GREETINGS = (
    "hi",
    "hello",
    "yo",
    "hey"
)

CLOSING = (
    "bye",
    "thanks",
    "thank you",
    "goodbye"
)

class ResponseType(StrEnum):
    Q = auto()
    A = auto()


def formulate_question(query: str) -> str:
    """
    Formulates a natural language question based on which facts are missing from the DB.
    """
    if 'prioritizedOver' in query:
        return "that depends on what you find important. What do you prioritize"
    elif 'hasPhysicalActivityHabit' in query:
        return "what physical activities do you regularly do"
    raise ValueError(f"Cannot formulate question for query {query}")

# =============================================================================
# def formulate_advice(activity: str) -> str:
#     prefix = "http://www.semanticweb.org/aledpro/ontologies/2024/2/userKG#"
#     activity = activity.replace(prefix, "")
# 
#     activity = activity.replace("_", " ")
#     return activity
# =============================================================================


def formulate_advice(activity: str) -> str:
    prefix = "http://www.semanticweb.org/aledpro/ontologies/2024/2/userKG#"
    activity = activity.replace(prefix, "")

    # Split activity on underscore and take the last part if it starts with "activity"
    parts = activity.split("_")
    if parts[0] == "activity":
        activity = "_".join(parts[1:])

    activity = activity.replace("_", " ")
    return activity


def generate_response(reasoner_response):
    sentence_data = reasoner_response['sentence_data']
    try:
        name = sentence_data['patient_name']
    except KeyError:
        name = "Unknown patient"
    current_app.logger.debug(f"reasoner_response: {reasoner_response}")
    response_type = ResponseType(reasoner_response["type"])
    response_data = reasoner_response["data"]

    message = "I don't understand, could you try rephrasing it?"

    if sentence_data['sentence'].lower() in GREETINGS:
        message = f"Hi, {name}"

    elif sentence_data['sentence'].lower() in CLOSING:
        message = f"Goodbye {name}"

    elif response_type == ResponseType.Q:
        if 'prioritizedOver' in response_data['data']:
            message = generate(f"The user talking to you is {name}. {name} is a diabetes patient. You currently know too little about {name} to help them. In particular, you would like to get to know what {name}'s general values are and how they prioritize them, so you can recommend activities to {name} that involve their values.", sentence_data['sentence'])

    elif response_type == ResponseType.A:
        activity = formulate_advice(response_data['data'][1])
        message = generate(f"The user talking to you is {name}. {name} is a diabetes patient. You know a little bit about {name} to help them. Based on their preferences and current situation, you propose the activity {activity}. Explain why it helps them combat diabetes.", sentence_data['sentence'])
        # message = f"How about the activity '{activity}', {name}?"

    return message


def send_message(reasoner_response):
    message = generate_response(reasoner_response)
    payload = {"message": message}
    current_app.logger.debug(f"sending response message: {payload}")
    front_end_address = current_app.config.get("FRONTEND_ADDRESS", None)
    if front_end_address:
        requests.post(f"http://{front_end_address}/process", json=payload)

