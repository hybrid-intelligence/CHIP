
import torch
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay



def evaluate_on_test_set(model, test_loader, device):
    model.eval()
    results_file = 'C:/Users/ntanavarass/Desktop/Conversational-Triple-Extraction-1/models/evaluation_results.csv'

    # Open a file to write the evaluation results
    with torch.no_grad(), open(results_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['token', 'sentence_id', 'predicted_label', 'gold_label'])

        all_preds = []
        all_labels = []
        all_tokens = []
        all_sentence_ids = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sentence_ids = batch['sentence_ids'].to(device)  

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Flatten arrays to process them easily
            preds = preds.view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()
            input_ids = input_ids.view(-1).cpu().numpy()
            sentence_ids = sentence_ids.view(-1).cpu().numpy()

            # Print lengths before applying the mask
            print(f'Length of input_ids: {len(input_ids)}')
            print(f'Length of labels: {len(labels)}')
            print(f'Length of sentence_ids: {len(sentence_ids)}')

            # Apply a mask to filter out padding tokens
            mask = labels != -100
            filtered_preds = preds[mask]
            filtered_labels = labels[mask]
            filtered_tokens = input_ids[mask]
            filtered_sentence_ids = sentence_ids[mask]

            # Print lengths after applying the mask
            print(f'Filtered length of input_ids: {len(filtered_tokens)}')
            print(f'Filtered length of labels: {len(filtered_labels)}')
            print(f'Filtered length of sentence_ids: {len(filtered_sentence_ids)}')

            all_preds.extend(filtered_preds)
            all_labels.extend(filtered_labels)
            all_tokens.extend(filtered_tokens)
            all_sentence_ids.extend(filtered_sentence_ids)

            # Write results to the CSV
            token_texts = [test_loader.dataset.tokenizer.convert_ids_to_tokens([id]) for id in filtered_tokens]
            for token, sid, pred, label in zip(token_texts, filtered_sentence_ids, filtered_preds, filtered_labels):
                predicted_label = list(test_loader.dataset.label_dict.keys())[list(test_loader.dataset.label_dict.values()).index(pred)]
                actual_label = list(test_loader.dataset.label_dict.keys())[list(test_loader.dataset.label_dict.values()).index(label)]
                writer.writerow([token[0], sid, predicted_label, actual_label])  

        # Compute and display metrics outside the file-writing block
        cm = confusion_matrix(all_labels, all_preds, labels=list(test_loader.dataset.label_dict.values()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_loader.dataset.label_dict.keys()))
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        disp.plot(cmap=plt.cm.Blues)
        plt.show()

        return precision, recall, f1
