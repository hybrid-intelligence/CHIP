
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup



class SPODataset(Dataset):
    """
    Manages and prepares the data for training BERT for SPO token-level labeling. 
    
    """
    def __init__(self, dataframe, tokenizer, max_len=512, stride=300):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.label_dict = self.create_label_dict()
        self.items = self.create_items()


    def create_label_dict(self):
        """
        Created a dictionary that mapp the SPO-labels (textual labels) to numeric indices.

        """
        unique_labels = ['Subject', 'Predicate', 'Object', 'other']

        return {label: idx for idx, label in enumerate(unique_labels)}


    def align_labels_and_sentence_ids_with_tokens(self, original_sentences, labels, sentence_ids):
        """
        Aligns the SPO-labels and sentence ids with their respective tokens after tokenization.
        
        """
        # Tokenize the input sentences
        tokenized_inputs = self.tokenizer(original_sentences, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=False, max_length=self.max_len)
        aligned_labels = []
        aligned_sentence_ids = []

        previous_word_idx = None  # Track the index of the last non-subtoken word

        for i, token in enumerate(tokenized_inputs.tokens()):
            word_idx = tokenized_inputs.word_ids()[i]  # Word index for the token

            # Assign labels
            if word_idx is not None:
                if word_idx != previous_word_idx:
                    # It's the start of a new word
                    aligned_label = self.label_dict[labels[word_idx]]
                    aligned_sentence_id = sentence_ids[word_idx]
                    previous_word_idx = word_idx
                else:
                    # Subsequent subtokens get dummy values
                    aligned_label = -100  # Dummy label for subtokens
                    aligned_sentence_id = -1  # Dummy sentence ID for subtokens
            else:
                # Special tokens like [CLS], [SEP], [PAD]
                aligned_label = -100
                aligned_sentence_id = -1

            aligned_labels.append(aligned_label)
            aligned_sentence_ids.append(aligned_sentence_id)

        return tokenized_inputs, aligned_labels, aligned_sentence_ids
    

    def create_items(self):
        """
        Prepares tokenized data into manageable pieces (sliding windows) for training.

        """
        items = []
        all_tokens = []
        all_labels = []
        all_sentence_ids = []

        # Flatten all tokens and labels into single lists
        for _, row in self.data.iterrows():
            all_tokens.append(row['token'])  
            all_labels.append(row['spo_label']) 
            all_sentence_ids.append(row['sentence_id']) 

        # Tokenize the entire sequence of tokens and align both labels and sentence IDs
        tokenized_input, aligned_labels, aligned_sentence_ids = self.align_labels_and_sentence_ids_with_tokens(all_tokens, all_labels, all_sentence_ids)
        
        # Ensure sliding window covers the entire sequence
        total_tokens = len(tokenized_input['input_ids'])
        print('The total tokens are:', total_tokens)

        # Apply sliding window technique
        start_index = 0

        while start_index < total_tokens:
            # Define the end index for the window
            end_index = min(start_index + self.max_len, total_tokens)
            
            # Extract the token sequence and label sequence for this window
            segment_input_ids = tokenized_input['input_ids'][start_index:end_index]
            segment_attention_mask = tokenized_input['attention_mask'][start_index:end_index]
            segment_labels = aligned_labels[start_index:end_index]
            segment_sentence_ids = aligned_sentence_ids[start_index:end_index]

            # Print the tokens and labels for this window
            print(f"Tokens for window starting at {start_index}: {self.tokenizer.convert_ids_to_tokens(segment_input_ids)}")
            print(f"Labels for window starting at {start_index}: {segment_labels}")
            print("=" * 80)  # Separator for better readability
            
            # Append the sequence to the items list
            items.append({
                'input_ids': torch.tensor(segment_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(segment_attention_mask, dtype=torch.long),
                'labels': torch.tensor(segment_labels, dtype=torch.long),
                'sentence_ids': torch.tensor(segment_sentence_ids, dtype=torch.long)
            })
            
            # Move the sliding window by the stride
            start_index += self.stride

        return items


    def __getitem__(self, index):
        return self.items[index]


    def __len__(self):
        return len(self.items) # The length of items represents the total sliding windows created
    

def collate_fn(batch):
    """
    Prepares batches by padding input_ids, attention masks, and labels to the same length for batch processing. It uses 'pad_sequence' to align sequences 
    in the batch to the longest sequence and returns a dictionary containing padded 'input_ids', 'attention_mask', and 'labels' ready for model input.
    
    """
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def train_model(model, train_loader, device, optimizer, scheduler):
    """ 
    Retrain BERT for token-level SPO-label classification from conversational sentences.

    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Calculate loss
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

        # Update weights
        optimizer.step()

        # Adjusting the learning rate using a scheduler
        scheduler.step()
        
        # Accumulate loss
        total_loss += loss.item()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds = preds.view(-1).cpu().numpy()
        labels = labels.view(-1).cpu().numpy()
        mask = labels != -100
        all_preds.extend(preds[mask])
        all_labels.extend(labels[mask])

    train_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(train_loader)

    return avg_loss, train_f1


def evaluate_model(model, val_loader, device):
    # Set the model to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            preds = preds.view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()
            mask = labels != -100     # Exclude the padding tokens for the f1-score calculation
            all_preds.extend(preds[mask])
            all_labels.extend(labels[mask])

    return f1_score(all_labels, all_preds, average='macro')


def train_and_evaluate_with_early_stopping(model, train_loader, val_loader, device, optimizer, scheduler, patience=2):

    # 'patience' is the number of epochs to continue training without improvement in the f1-score before stopping. 

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(10):  # up to 10 epochs
        avg_train_loss, train_f1 = train_model(model, train_loader, device, optimizer, scheduler)
        val_f1 = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping logic based on validation f1-score
        if val_f1  > best_f1:
            best_f1 = val_f1 
            patience_counter = 0
            print(f"New best F1: {best_f1:.4f}, saving model...")
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            print(f"No improvement, patience counter: {patience_counter}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    model.load_state_dict(torch.load("best_model.pt"))

    return best_f1


def main(trial):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 8, 16])
    
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(training.label_dict))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    try:
        best_f1 = train_and_evaluate_with_early_stopping(model, train_loader, val_loader, device, optimizer, scheduler)
        print(f"Returning F1 Score: {best_f1}")
        return best_f1
    except Exception as e:
        print(f"An error occurred in main: {e}")
        return 0  # Provide a default F1 score in case of error


def objective(trial):
    return main(trial)