
# Import the libraries
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from src.model_training import SPODataset, objective
from src.model_evaluation import evaluate_on_test_set
from src.forming_triples import form_triples
from src.evaluate_triples import evaluate_triples



# Load the datasets for training, validation, and testing
train_data = pd.read_csv('.../data/training_data.csv')
validation_data = pd.read_csv('.../data/validation_data.csv')
test_data = pd.read_csv('.../data/testing_data.csv')

# Initialize the tokenizer for BERT model preprocessing
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Prepare the datasets for BERT
training = SPODataset(train_data, tokenizer)
validation = SPODataset(validation_data, tokenizer)
testing = SPODataset(test_data, tokenizer)

# Train the model / Hyperparameter tuning
study = optuna.create_study(direction='maximize')  # Find hyperparameters that yield the highest value of the objective function
study.optimize(objective, n_trials=10)  # Perform 10 trials to find the best hyperparameters

# Load the best-performing model based on the Optuna study
best_model_path = 'C:/Users/ntanavarass/Desktop/Conversational-Triple-Extraction-1/models/best_model.pt'
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(testing.label_dict))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)

# Prepare the DataLoader for the test set
test_loader = DataLoader(testing, batch_size=1, shuffle=False)

# Compute precision, recall, and F1 score by evaluating the model on the test set
precision, recall, f1 = evaluate_on_test_set(model, test_loader, device)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Load the results of the model evaluation
evaluation_results = pd.read_csv('.../results/evaluation_results.csv')

# Generate triples from the evaluated tokens by combining them based on their predicted labels
form_triples(evaluation_results)

# Save the conversational sentences along with their triples
evaluation_results.to_csv('.../results/bert_triples.csv', index=False)

# Evaluate the quality of the extracted triples
triples_evaluation = evaluate_triples(evaluation_results)
print(triples_evaluation)