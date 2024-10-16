
from ast import literal_eval



def calculate_f1_scores(matches, total_gold, total_pred):
    # Calculate precision, recall, and F1 score manually
    precision = sum(matches) / total_pred if total_pred else 0
    recall = sum(matches) / total_gold if total_gold else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def evaluate_triples(data):
    # Ensure tuples are correctly handled
    data['predicted_triple'] = data['predicted_triple'].apply(lambda x: x if isinstance(x, tuple) else literal_eval(x))
    data['gold_triple'] = data['gold_triple'].apply(lambda x: x if isinstance(x, tuple) else literal_eval(x))

    # Initialize match lists
    full_matches = []
    partial_matches = []
    subject_matches = []
    predicate_matches = []
    object_matches = []

    # Iterate over each row in the dataframe to evaluate matches
    for _, row in data.iterrows():
        gold = row['gold_triple']
        pred = row['predicted_triple']

        # Check for cases where either gold or predicted is completely empty
        if gold == ('', '', ''):
            if pred == ('', '', ''):
                # Correct identification of no triple
                full_matches.append(True)
                partial_matches.append(True)
                subject_matches.append(True)
                predicate_matches.append(True)
                object_matches.append(True)
            else:
                # False positive: predicted a triple where there should be none
                full_matches.append(False)
                partial_matches.append(False)
                subject_matches.append(False)
                predicate_matches.append(False)
                object_matches.append(False)
        elif pred == ('', '', ''):
            # False negative: failed to predict a triple that should exist
            full_matches.append(False)
            partial_matches.append(False)
            subject_matches.append(False)
            predicate_matches.append(False)
            object_matches.append(False)
        else:
            # Evaluate non-empty triples
            full_match = gold == pred
            full_matches.append(full_match)
            
            partial_match = False
            component_match = [False, False, False]
            for i, (g_comp, p_comp) in enumerate(zip(gold, pred)):
                if g_comp == p_comp:
                    partial_match = True
                    component_match[i] = True

            partial_matches.append(partial_match)
            subject_matches.append(component_match[0])
            predicate_matches.append(component_match[1])
            object_matches.append(component_match[2])

    # Calculate precision, recall, F1 scores 
    full_match_precision, full_match_recall, full_match_f1 = calculate_f1_scores(full_matches, len(data), len(data))
    partial_match_precision, partial_match_recall, partial_match_f1 = calculate_f1_scores(partial_matches, len(data), len(data))
    subject_precision, subject_recall, subject_f1 = calculate_f1_scores(subject_matches, len(data), len(data))
    predicate_precision, predicate_recall, predicate_f1 = calculate_f1_scores(predicate_matches, len(data), len(data))
    object_precision, object_recall, object_f1 = calculate_f1_scores(object_matches, len(data), len(data))

    return {
        "full_match": 
        {"precision": full_match_precision, "recall": full_match_recall, "f1": full_match_f1},
        "partial_match": 
        {"precision": partial_match_precision, "recall": partial_match_recall, "f1": partial_match_f1},
        "subject_match": 
        {"precision": subject_precision, "recall": subject_recall, "f1": subject_f1},
        "predicate_match": 
        {"precision": predicate_precision, "recall": predicate_recall, "f1": predicate_f1},
        "object_match": 
        {"precision": object_precision, "recall": object_recall, "f1": object_f1}
    }
