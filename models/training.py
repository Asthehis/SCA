import json
import pandas as pd
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# on charge le fichier contenant le dataset pour entraîner le modèle 
data = []
with open("training_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        # Conversion label en binaire
        label = 1 if obj['label'].lower() == "affirmative." else 0
        # Construire un texte combiné (tu peux ajuster)
        text = obj['context'].replace('\n', ' ') + " | " + obj['keyword']
        data.append({"text": text, "label": label})

df = pd.DataFrame(data)
# print(df.head())

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",  # Sauvegarde après chaque epoch
)

# on sépare le dataset en train et test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Puis encode les deux :
encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_test = test_dataset.map(preprocess_function, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test,
    compute_metrics=compute_metrics  
)

trainer.train()

model.save_pretrained("./camembert_custom_model")
tokenizer.save_pretrained("./camembert_custom_model")
