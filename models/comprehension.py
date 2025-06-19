import json
import re
import os
from gpt4all import GPT4All

# === Chargement du modèle GPT4All ===
model_filename = "ggml-gpt4all-j-v1.3-groovy.bin"
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # dossier parent = racine du projet

# Vérification du fichier
full_path = os.path.join(model_dir, model_filename)
if not os.path.exists(full_path):
    raise FileNotFoundError(f"Modèle non trouvé à : {full_path}")

# Chargement du modèle
model = GPT4All(model_filename, model_path=model_dir)
model.open()

# === Fonctions utilitaires ===

def load_keywords(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("keywords", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return []


def load_transcript(file_path):
    transcript = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if " : " in line:
                    speaker, text = line.strip().split(" : ", 1)
                    transcript.append({"speaker": speaker, "text": text})
    except FileNotFoundError:
        print(f"Erreur : fichier {file_path} introuvable.")
    return transcript


def get_matched_keywords(text, keyword_entry):
    keywords = [keyword_entry.get("word", "")] + keyword_entry.get("synonyms", [])
    return [word for word in keywords if word and re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)]


def is_positive_response(context, keyword):
    prompt = (
        f"Voici un extrait de conversation contenant le mot '{keyword}':\n"
        f"{context}\n"
        f"Le mot '{keyword}' est-il utilisé ici de manière affirmative ou négative ? "
        f"Répondez strictement par 'affirmative' ou 'négative'."
    )

    response = model.prompt(prompt, max_tokens=30)
    print(f"\n🧠 Contexte analysé :\n{context}")
    print(f"🔍 Réponse locale : {response.strip()}\n")

    return "affirmative" in response.lower()


def analyze_transcript(transcript, keywords):
    validated_words = {}

    for i, line in enumerate(transcript):
        for entry in keywords:
            base_word = entry.get("word", "")
            matched_words = get_matched_keywords(line["text"], entry)

            for matched in matched_words:
                print(f"✅ Mot trouvé : '{matched}' (lié à '{base_word}') à la ligne {i + 1}")
                # Contexte local (2 lignes avant et 2 après)
                start = max(i - 2, 0)
                end = min(i + 3, len(transcript))
                context_lines = [f"{l['speaker']} : {l['text']}" for l in transcript[start:end]]
                context = "\n".join(context_lines)

                if is_positive_response(context, matched):
                    validated_words[base_word] = entry.get("severity", "N/A")
                else:
                    print(f"❌ Négation ou incertitude pour '{matched}'")

    return validated_words


def save_validated_words(validated_words, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for word, severity in validated_words.items():
            file.write(f"{word} (Sévérité: {severity})\n")


# === Script principal ===
if __name__ == "__main__":
    sca_keywords = load_keywords("data/sca_words.json")
    non_sca_keywords = load_keywords("data/non_sca_words.json")

    try:
        with open("data/last_filename.txt", "r", encoding="utf-8") as f:
            filename = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("Fichier 'last_filename.txt' introuvable.")

    transcript_path = f"data/raw/{filename.replace('.m4a', '_diarized.txt')}"
    transcript = load_transcript(transcript_path)

    print("📑 Analyse des mots-clés SCA...")
    validated_sca = analyze_transcript(transcript, sca_keywords)

    print("\n📑 Analyse des mots-clés NON-SCA...")
    validated_non_sca = analyze_transcript(transcript, non_sca_keywords)

    save_validated_words(validated_sca, "mots_dits_sca.txt")
    save_validated_words(validated_non_sca, "mots_dits_non_sca.txt")

    print("\n✅ Mots-clés validés enregistrés dans les fichiers de sortie.")
