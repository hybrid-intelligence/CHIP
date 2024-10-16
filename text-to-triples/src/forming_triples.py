

def form_triples(data):
    """ 
    Extracts SPO triples based on predicted and gold labels for each token. Groups the data by sentence and 
    forms triples by concatenating tokens classified as 'Subject', 'Predicate', and 'Object'.
    
    """
    # Group data by sentences to process each sentence individually
    grouped = data.groupby('sentence_id')

    # Prepare a list to store the triples
    predicted_triples = []
    gold_triples = []

    for sentence_id, group in grouped:
        # Extract tokens by their predicted labels
        predicted_subject_tokens = ' '.join(group[group['predicted_label'] == 'Subject']['token'])
        predicted_predicate_tokens = ' '.join(group[group['predicted_label'] == 'Predicate']['token'])
        predicted_object_tokens = ' '.join(group[group['predicted_label'] == 'Object']['token'])

        # Extract tokens by their gold labels
        gold_subject_tokens = ' '.join(group[group['gold_label'] == 'Subject']['token'])
        gold_predicate_tokens = ' '.join(group[group['gold_label'] == 'Predicate']['token'])
        gold_object_tokens = ' '.join(group[group['gold_label'] == 'Object']['token'])

        # Form the triples
        predicted_triple = (predicted_subject_tokens, predicted_predicate_tokens, predicted_object_tokens)
        gold_triple = (gold_subject_tokens, gold_predicate_tokens, gold_object_tokens)

        # Extend the triples for each token in the sentence
        predicted_triples.extend([predicted_triple] * len(group))
        gold_triples.extend([gold_triple] * len(group))

    # Add the triples as new columns to the original dataframe
    data['predicted_triple'] = predicted_triples
    data['gold_triple'] = gold_triples
