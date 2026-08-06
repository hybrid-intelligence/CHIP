# This is a CHIP Module
| Properties    |                     |
| ------------- | -------------       |
| **Name**      | Stergios Triple Extractor (placeholder) |
| **Type**      | Triple Extractor  |
| **Core**      | Yes |
| **Access URL**       | N/A |

## Description
Placeholder triple extractor module. It exposes the standard `/process` route and forwards a payload with an empty `triples` list to the reasoner. Replace the logic in `app/util/__init__.py::extract_triples` with the actual extraction implementation.

## Usage
Instructions:
1. Configure `core-modules.yaml` to use this module as the triple extractor.

## Input/Output
Communication between the core modules occurs by sending a POST request to the `/process` route with an appropriate body, as detailed below.

### Input from `Front-End`
```JSON
{
    "patient_name": <string>,   // The name of the user currently chatting
    "sentence": <string>,       // The sentence that the user submitted
    "timestamp": <string>       // The time at which the user submitted the sentence (ISO format)
}
```

### Output to `Reasoner`
```JSON
{
    "sentence_data": {
        "patient_name": <string>,   // The name of the user currently chatting
        "sentence": <string>,       // The sentence that the user submitted
        "timestamp": <string>       // The time at which the user submitted the sentence (ISO format)
    },
    "triples": [
        {
            "subject":<string>, 
            "object": <string>, 
            "predicate":<string>
        },
        ...
    ]
}
```

## API (routes, descriptions, models)
- [GET] `/`: default 'hello' route, to check whether the module is alive and kicking.
- [POST] `/process`: accepts sentence data and forwards extracted triples to the reasoner.

## Internal Dependencies
None.

## Required Resources
None.
