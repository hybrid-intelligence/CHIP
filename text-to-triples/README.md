# Conversational Triple Extraction in Diabetes Healthcare Management Using Synthetic Data

## Project Overview

This document describes the process of fine-tuning a BERT model, to classify tokens within conversational sentences according to SPO (Subject, Predicate, Object) labels. Following the classification, the predicted labels are used to format triples accordingly.


## Dependencies

- Before running the code, ensure you have Python 3.12.3 installed on your system as it is required for compatibility with the libraries used in this project.
  
- Make sure you have the required Python libraries installed. You can install them using the following command:

```bash
pip install tensorflow keras torch torchvision scikit-learn numpy pandas matplotlib transformers optuna
```

## Scripts Overview

1) ```model_training.py```:
   - This script is responsible for finetuning a pretrained BERT model (```BERT-base-uncased```) for token-level SPO-label classification from conversational sentences. It includes data preparation, model training, and evaluation functions. The ```SPODataset``` class within handles the data management, ensuring tokens are appropriately processed and aligned with their respective SPO labels. The main function orchestrates the loading of data, model instantiation, training, and evaluation, allowing for experimentation with different hyperparameters through Optuna optimization.

2) ```model_evaluation.py```:
   - This script is designed to evaluate the performance of the BERT model on a token-level classification task using a test dataset.
  
3) ```forming_triples.py```:
   - This script focuses on the extraction of SPO (Subject, Predicate, Object) triples from the evaluation results that contain both predicted and gold labels for each token.

4) ```evaluate_triples.py```:
   - This script is about evaluating the quality of extracted SPO triples by comparing predicted triples against gold triples. 
  
5) ```main.py```:
   - This comprehensive script orchestrates the full workflow of training, evaluating, and extracting triples from conversational data using a BERT model. 


## How to Use

1) Before running the scripts, ensure all required libraries and dependencies are installed in your Python environment.
2) Open the ```main.py``` file and update the dataset paths to match the locations where your data files are stored on your system.
3) In the ```main.py``` file, specify the paths for saving the evaluation results and extracted triples. Double-check these paths to ensure they are correct to prevent any file-not-found errors during execution.
4) Run the ```main.py``` file in your Python environment (Visual Studio Code is recommended). 
5) Once the script has executed successfully, you can find the evaluation results and extracted triples in the specified output folder, typically located at ```.../results/```.





