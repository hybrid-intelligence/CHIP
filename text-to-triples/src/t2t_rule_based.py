import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



def extract_triples(patients_data):
    triples = []
    for patient in patients_data:
        patient_name = patient["patient_name"]
        for sentence in patient["sentences"]:
            tokens = word_tokenize(sentence)
            tagged_tokens = pos_tag(tokens)

            subject = patient_name  
            predicate = None
            object_ = None
            for word, tag in tagged_tokens:
                if tag.startswith('VB') and predicate is None:
                    predicate = word
                elif tag.startswith(('NN', 'NNS', 'NNP', 'NNPS')) and predicate and subject and object_ is None:
                    object_ = word
                    break

            if subject and predicate and object_:
                triple_dict = {"subject": subject, "predicate": predicate, "object": object_}
                triples.append(triple_dict)

    return {"triples": triples}


def main():
    file_path = 'C:/Users/ntanavarass/Desktop/chip-demo/text-to-triples/src/patient_conversations.json' # specify the absolute path to the directory containing patient conversations
    with open(file_path, 'r') as file:
        conversations = json.load(file)
    
    result = extract_triples(conversations)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
