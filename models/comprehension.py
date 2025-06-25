import json
import re
import os
from pathlib import Path
from llama_cpp import Llama 

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"  
USE_MISTRAL_FOR_LABELS = True  # permet de labeliser le dataset automatiquement

# Fonctions utilitaires
def load_keywords(file_path):
    """ 
    Cette fonction permet d'ouvrir les fichiers .json contenant les mots-clés SCA et non SCA, et d'en récupérer les données.
    Retourne une liste composé de dictionnaire, chaque dic représente un mot-clé. Celui-ci est associé à ses synonymes, ses triggers et sa sévérité.

    - file_path : le chemin d'accès au fichier
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("keywords", []) # On récupère toutes les infos sous la balise "keywords"
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return [] # Affichage d'un message d'erreur et renvoie d'une liste vide si le fichier n'est pas trouvé


def load_transcript(file_path):
    """
    Cette fonction permet d'ouvrir le fichier .txt contenant la transcription de l'audio à analyser.
    Retourne une liste de dictionnaire, chaque dic est une phrase prononcée, avec le timecode et le texte. 

    - file_path : le chemin d'accès au fichier
    """
    transcript = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file: # On parcourt chaque ligne du fichier
                if " : " in line: # Le texte est sous ce format : "0.03s - 5.48s: Bonjour monsieur"
                    speaker, text = line.strip().split(" : ", 1) # On peut donc récupérer le timecode à gauche et le texte à droite du symbole ':'
                    transcript.append({"speaker": speaker, "text": text}) # A modif ici car on utilise plus le speaker
    except FileNotFoundError:
        print(f"Erreur : fichier {file_path} introuvable.") # Affichage d'un message d'erreur si le fichier n'est pas trouvé
    return transcript


def get_keywords_in_text(text, entry):
    """
    Cette fonction permet de trouver les mots clés dans le texte.
    Elle renvoie une liste en compréhension.

    - text : la transcription où l'on cherche les mots clés
    - entry : la base de données de mots clés
    """
    all_words = [entry["word"]] + entry.get("synonyms", [])
    return [
        w for w in all_words
        if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)
    ]


def format_context(transcript, start_idx, window=2):
    """
    """
    start = max(start_idx - window, 0)
    end = min(start_idx + window + 1, len(transcript))
    return " ".join(f"{line['speaker']} : {line['text']}" for line in transcript[start:end])


def ask_mistral(context, keyword, model):
    """
    """
    prompt = (
        f"Voici un extrait de conversation contenant le mot '{keyword}':\n"
        f"{context}\n"
        f"Le mot '{keyword}' est-il utilisé ici de manière affirmative (positive) ou négative ?\n"
        f"Répondez strictement par 'affirmative' ou 'négative'. Un seul mot. Aucune explication.\n"
        f"Réponse :"
    )
    result = model(prompt, max_tokens=5)
    return result["choices"][0]["text"].strip().lower()


def generate_dataset(transcript_path, keywords_path, output_jsonl):
    """
    """
    transcript = load_transcript(transcript_path)
    keywords = load_keywords(keywords_path)
    dataset = []

    model = None
    if USE_MISTRAL_FOR_LABELS:
        model = Llama(model_path=MODEL_PATH, verbose=False, n_ctx=32768)

    for i, line in enumerate(transcript):
        for entry in keywords:
            matched = get_keywords_in_text(line["text"], entry)
            for kw in matched:
                context = format_context(transcript, i)
                label = None
                if USE_MISTRAL_FOR_LABELS:
                    answer = ask_mistral(context, kw, model)
                    if answer in ["affirmative", "négative", "affirmative.", "négative."]:
                        label = answer
                    else:
                        label = "ambigue"
                else:
                    label = "TODO"

                dataset.append({
                    "context": context,
                    "keyword": kw,
                    "label": label
                })

    with open(output_jsonl, "a", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{len(dataset)} exemples écrits dans {output_jsonl}")


# main
if __name__ == "__main__":
    transcript_dir = "data/transcript"
    keyword_path = "data/sca_non_sca_words.json"
    output_path = "training_data.jsonl"
    for file_name in os.listdir(transcript_dir):
        if file_name.lower().endswith(".txt"):
            transcript_path = os.path.join(transcript_dir, file_name)
            generate_dataset(transcript_path=transcript_path, keywords_path=keyword_path, output_jsonl=output_path)
