# Conversational Triple Extraction in Diabetes Healthcare Management Using Synthetic Data

## Overview

This document describes the procedure for using a fine-tuned BERT model to extract S-P-O (Subject-Predicate-Object) triples from conversational sentences related to Diabetes management.


## Dependencies

- Before running the code, ensure you have Python 3.12.3 installed on your system as it is required for compatibility with the libraries used in this project.
  
- Make sure you have the required Python libraries installed. You can install them using the following command:

```bash
pip install torch transformers
```

## Script Overview

- ```t2t_bert.py```: This script utilizes a fine-tuned BERT model to extract Subject, Predicate, and Object (S-P-O) triples from conversational sentences. It loads the fine-tuned BERT model and uses a tokenizer to process input sentences. The script includes functions to predict S-P-O labels at a token level and assemble these tokens into coherent triples, handling cases where some components may be implied. If no explicit components are identified, it returns an empty triple structure.


## How to Use

1) Before running the scripts, ensure all required libraries and dependencies are installed in your Python environment.
2) Access the fine-tuned BERT model by visiting https://huggingface.co/StergiosNt/spo_labeling_bert and downloading the model.
3) Open the ```t2t_bert.py``` file and update the path to where you have stored the downloaded fine-tuned BERT model.
4) Run the ```t2t_bert.py``` file in your Python environment (Visual Studio Code is recommended). 






