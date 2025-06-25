import json
import pandas as pd
from datasets import Dataset

# Charger ton fichier JSON ligne par ligne
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
print(df.head())


dataset = Dataset.from_pandas(df)

