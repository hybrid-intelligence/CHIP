import torch
from transformers import BertTokenizerFast
import json



# Load the finetuned BERT model from the specified file path
# Replace '<path_to_model>' with the actual path to the model file
model_path = r'<path_to_model>/best_model.pth'
model = torch.load(model_path)

# Switch the model to evaluation mode
model.eval()

# Initialize the tokenizer for processing inputs
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define the label map according to the training labels
label_map = {
    'LABEL_0': 'Subject',
    'LABEL_1': 'Predicate',
    'LABEL_2': 'Object',
    'LABEL_3': 'other'  
}

def predict_and_form_triples(input_data, model, tokenizer, label_map):
    """ 
    Extract and form S-P-O triples (Subject, Predicate, Object) from a sentence.
    This function tokenizes the sentence, uses a model to predict SPO labels for each token, and aggregates 
    these tokens into coherent phrases. It handles subwords by merging them with preceding tokens and forms 
    triples even if some components are implied. If all components are missing, it returns an empty triple.

    Args:
        input_data (dict): Contains the sentence to be processed.
        model (torch.nn.Module): Pre-trained model for S-P-O prediction.
        tokenizer (BertTokenizerFast): Tokenizer for processing the input sentence.
        label_map (dict): Maps label IDs to their descriptive labels (e.g., Subject, Predicate, Object, other).

    Returns:
        dict: Contains formed triples or an empty structure if no explicit S-P-O components are found.
    """

    sentence = input_data['sentence']
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    subjects, predicates, objects = [], [], []
    aggregated_subjects, aggregated_predicates, aggregated_objects = '', '', ''
    for token, pred in zip(tokens, predictions):
        label = label_map[model.config.id2label[pred.item()]] 
        
        # Handle subwords and aggregate them into the previous token if necessary
        if token.startswith("##"):
            if label == 'Subject' and subjects:
                subjects[-1] += token[2:]  # Remove '##' and concatenate
            elif label == 'Predicate' and predicates:
                predicates[-1] += token[2:]
            elif label == 'Object' and objects:
                objects[-1] += token[2:]
        else:
            if label == 'Subject':
                subjects.append(token)
            elif label == 'Predicate':
                predicates.append(token)
            elif label == 'Object':
                objects.append(token)

    # Join tokens to form the phrases for subjects, predicates, and objects
    aggregated_subjects = " ".join(subjects)
    aggregated_predicates = " ".join(predicates)
    aggregated_objects = " ".join(objects)

    triples = []
    # Form a triple only if there's at least one non-implied component
    if subjects or predicates or objects:
        triples.append({
            "subject": aggregated_subjects,
            "predicate": aggregated_predicates,
            "object": aggregated_objects
        })
    # If no components are available (all implied), then return an empty triple structure
    else:
        triples.append({
            "subject": "",
            "predicate": "",
            "object": ""
        })

    return {"triples": triples}


def process_input_output(input_json):
    """ 
    Processes input JSON string to extract S-P-O triples using the 'predict_and_form_triples' function. 
    It converts the input JSON string to a dictionary, applies the S-P-O prediction and aggregation logic, 
    and then serializes the result back into a JSON string with formatted output.

    Args:
        input_json (str): A JSON string containing the input data.

    Returns:
        str: A JSON string formatted to include S-P-O triples extracted from the input, or an empty structure if no components are explicit.
    """

    input_data = json.loads(input_json)
    result = predict_and_form_triples(input_data, model, tokenizer, label_map)
    return json.dumps(result, indent=4)


# Example usage
input_json = json.dumps({
    "patient_name": "John",
    "sentence": "I have started using nicotine gum.",
    "timestamp": "2024-10-24T10:00:00Z"
})

output_json = process_input_output(input_json)
print(output_json)