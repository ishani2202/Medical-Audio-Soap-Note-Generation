# README: Medical SOAP Detection and Classification

## Overview

This project is designed to process medical dialogue datasets and classify text into SOAP (Subjective, Objective, Assessment, Plan) categories. It utilizes NLP tools, transformers, and entity recognition models to analyze doctor-patient interactions and extract structured medical information.

---

## Requirements

### Libraries and Tools
- Python 3.x
- Transformers Library (`transformers`)
- Datasets Library (`datasets`)
- Pandas
- spaCy
- PyTorch (for transformer models)
- Pre-trained Models:
  - `Falconsai/medical_summarization`
  - `blaze999/Medical-NER`

### Installation
Install the required libraries using the following commands:
```bash
pip install transformers datasets spacy pandas
python -m spacy download en_core_web_sm
```

---

## Data Source

The dataset is loaded from the Hugging Face dataset repository:
- **Dataset Name**: `Magneto/modified-medical-dialogue-soap-summary`

---

## Workflow

### 1. **Data Loading**
The dataset is fetched using the `datasets` library and preprocessed for analysis:
```python
ds = load_dataset("Magneto/modified-medical-dialogue-soap-summary")
```

### 2. **Named Entity Recognition (NER)**
A pre-trained NER model (`blaze999/Medical-NER`) is used to extract entities from the dataset.

### 3. **SOAP Categorization**
Entities are classified into SOAP categories:
- **Subjective (S):** Patient-reported symptoms or history.
- **Objective (O):** Observations and diagnostic data.
- **Assessment (A):** Clinical interpretations or diagnoses.
- **Plan (P):** Treatment and follow-up actions.

#### Key Steps:
1. **NER Pipeline Setup**:
   ```python
   tokenizer = AutoTokenizer.from_pretrained("blaze999/Medical-NER")
   model = AutoModelForTokenClassification.from_pretrained("blaze999/Medical-NER")
   ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
   ```
2. **Entity Classification**:
   Each entity is mapped to its respective SOAP category based on predefined word and entity mappings.

3. **Data Aggregation**:
   Entities are aggregated and stored in CSV files for further analysis.

---

## Outputs

1. **Filtered CSV Files**:
   - `S_entities.csv`: Subjective entities.
   - `O_entities.csv`: Objective entities.
   - `A_entities.csv`: Assessment entities.
   - `P_entities.csv`: Plan entities.

2. **Word and Entity Frequency Analysis**:
   Top 10 most frequent words and entities are identified for each SOAP category.

3. **Sentence Classification**:
   Sentences from the dataset are classified into their respective SOAP categories using word and entity mapping.

---

## Functions and Modules

### Key Functions
1. **Entity Extraction**:
   Extracts entities using the NER pipeline.
2. **SOAP Classification**:
   Classifies sentences into SOAP categories based on word and entity mappings.
3. **CSV Update**:
   Updates CSV files with new entities or creates them if not present.

### Custom Dictionaries
- **Word-to-Category Mapping**:
  Maps common words to SOAP categories.
- **Entity-to-Category Mapping**:
  Maps NER-detected entities to SOAP categories.

---

## Personalization

The notebook includes a custom classification algorithm that combines:
- Lexical matching with word mappings.
- Entity recognition using pre-trained models.
- Fallback strategies to resolve ambiguities between SOAP categories.

---

## Limitations and Improvements

### Current Limitations:
- Dependence on predefined word and entity mappings.
- Challenges in handling ambiguous sentences.
- Computational overhead for large datasets.

### Potential Improvements:
- Integrating contextual embeddings (e.g., BERT, BioBERT) for more accurate classification.
- Adding a visualization module for SOAP classification results.
- Incorporating cross-validation for entity classification.

---

## Conclusion

This project demonstrates a robust pipeline for analyzing medical dialogues and structuring data into SOAP notes. It provides a foundation for advanced medical NLP applications, including summarization and clinical decision support.

---

# README: Model Training Workflow

## Overview

This notebook implements a workflow for training machine learning models using the Hugging Face Transformers library. It focuses on fine-tuning a pre-trained model for a specific task and evaluating the model's performance.

---

## Requirements

### Libraries and Tools
- Python 3.x
- Hugging Face Transformers (`transformers`)
- PyTorch
- Pandas
- scikit-learn

### Installation
Install the required libraries using the following commands:
```bash
pip install transformers torch scikit-learn pandas
```

---

## Workflow

### 1. **Model Initialization**
A pre-trained model (e.g., T5 or similar transformer model) is loaded using the Hugging Face library. The tokenizer is also initialized for text preprocessing:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModelForSeq2SeqLM.from_pretrained("model_name")
```

### 2. **Data Preprocessing**
- Text inputs and outputs are tokenized with truncation and padding to ensure compatibility with the model's input size.
- Labels are prepared for the supervised learning task:
```python
inputs = tokenizer(batch["input"], truncation=True, padding="max_length", return_tensors="pt")
outputs = tokenizer(batch["output"], truncation=True, padding="max_length", return_tensors="pt")
```

### 3. **Training Configuration**
The `TrainingArguments` class is used to define the training configuration, including:
- Batch size
- Learning rate
- Number of epochs
- Evaluation strategy
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs"
)
```

### 4. **Metrics Calculation**
Custom metrics such as precision, recall, F1 score, and accuracy are computed using scikit-learn:
```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    logits = pred.predictions.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, logits, average="weighted")
    accuracy = accuracy_score(labels, logits)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
```

### 5. **Training and Evaluation**
The Hugging Face `Trainer` class is used for model training and evaluation:
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
```

---

## Outputs

1. **Evaluation Results**:
   Metrics such as precision, recall, F1 score, and accuracy are displayed.
2. **Trained Model**:
   The fine-tuned model is saved in the specified output directory (`./results`).

---

## Limitations and Improvements

### Current Limitations:
- Dependence on pre-defined evaluation metrics.
- Limited batch size due to memory constraints.

### Potential Improvements:
- Implement gradient accumulation for large batch sizes.
- Experiment with advanced optimizers and learning rate schedules.

---

## Conclusion

This workflow demonstrates the process of fine-tuning a pre-trained transformer model for a specific NLP task. It provides a robust foundation for extending this workflow to various other machine learning tasks.

---
