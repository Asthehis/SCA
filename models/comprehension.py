import json
import re
import os
from llama_cpp import Llama

model = Llama(model_path="models/mistral-7b-instruct-v0.2.Q5_K_M.gguf", verbose=False)


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


def get_matched_keywords(text, keyword_entry):
    """
    Cette fonction permet d'obtenir les mots-clés (et synonymes) présents dans une phrase. 
    Retourne les mots-clés détectés dans le texte.

    - text : le texte dans lequel on cherche les mots.
    - keyword_entry : un dictionnaire contenant :
        - "word" : le mot-clé principal
        - "synonyms" : une liste de synonymes (optionnelle)

    Exemple :
    keyword_entry = {"word": "homme", "synonyms": ["Mari", "frère", "fils", "époux", "compagnon", "père" ]}
    text = "Mon mari a mal à la poitrine."
    => retourne ["mari"]
    """
    keywords = [keyword_entry.get("word", "")] + keyword_entry.get("synonyms", []) # On fait une liste avec tous les mots clés et leurs synonymes
    return [ #Liste en compréhension : on garde uniquement les mots détectés dans le texte
        word for word in keywords 
        if word and re.search( # Expression pour rechercher le mot dans 'text' (ici la phrase)
            rf"\b{re.escape(word)}\b", # \b = délimiteur de mot entier (ex : dou ne matchera pas douleur)
            text, 
            re.IGNORECASE # ignore maj et min
        )
    ]



def is_positive_response(context, keyword):
    """
    Cette fonction permet de déterminer si une réponse est positive ou non.
    Elle renvoie un booléen. True si la réponse est affirmative, false sinon.

    -context : le contexte contenant le mot clé, la phrase où est détecté le mot et les 2 phrases avant et après.
    -keyword : le mot clé présent dans la phrase et en cours d'analyse
    """
    prompt = (
        f"Voici un extrait de conversation contenant le mot '{keyword}':\n"
        f"{context}\n"
        f"Le mot '{keyword}' est-il utilisé ici de manière affirmative (positive) ou négative ?\n"
        f"Répondez strictement par 'affirmative' ou 'négative'. Un seul mot. Aucune explication.\n"
        f"Réponse :"
    ) # On rédige un prompt à l'IA 

    response = model(prompt, max_tokens=5, stop=["\n"])  # LlamaCpp retourne un dictionnaire
    answer = response["choices"][0]["text"].strip().lower()

    print(f"\nContexte analysé :\n{context}")
    print(f"Réponse locale : {answer}\n")

    if "affirmative" in answer:
        return True
    elif "négative" in answer:
        return False
    else:
        print(f"Réponse ambiguë ou inattendue : {answer}")
        return False


def analyze_transcript(transcript, keywords):
    """
    Retourne le dic des mots clés valides.
    -transcript : la transcription de l'audio à analyser
    -keywords : les mots-clés
    """
    validated_words = {} # dic vide qui va être remplis par les mots-clés validés
    
    for i, line in enumerate(transcript): # transcript : liste de dic, enumerate : 
        # print("line:", line)
        # print(line["text"])
        for entry in keywords: # On parcourt les mots-clés
            base_word = entry.get("word", "")
            matched_words = get_matched_keywords(line["text"], entry) # On regarde si des mots clés match avec les mots dans une ligne

            for matched in matched_words: # On vérifie maintenant que les mots 'matché' sont valides ou non
                print(f"Mot trouvé : '{matched}' (lié à '{base_word}') à la ligne {i + 1}")
                # Contexte local (2 lignes avant et 2 après)
                start = max(i - 2, 0)
                end = min(i + 3, len(transcript))
                context_lines = [f"{l['speaker']} : {l['text']}" for l in transcript[start:end]]
                context = "\n".join(context_lines)

                if is_positive_response(context, matched): # Si le mots clés est valide alors on l'ajoute au dic
                    validated_words[base_word] = entry.get("severity", "N/A")

    return validated_words


def save_validated_words(validated_words, output_file):
    """
    Cette fonction permet de sauvegarder les mots-clés validés.
    Créé un fichier .txt avec les mots et leur sévérité.

    -validated_words : liste des mots-clés validés
    -output_file : le chemin où sera sauvegarder le fichier
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for word, severity in validated_words.items():
            file.write(f"{word} (Sévérité: {severity})\n")


# Script principal
if __name__ == "__main__":

    sca_keywords = load_keywords("data/sca_words.json")
    # print(sca_keywords)
    non_sca_keywords = load_keywords("data/non_sca_words.json")
    # print(non_sca_keywords)

    transcript_path = f"data/raw/Audio-SCA-1_diarized.txt"
    transcript = load_transcript(transcript_path)
    # print(transcript)

    print("Analyse des mots-clés SCA...")
    validated_sca = analyze_transcript(transcript, sca_keywords)

    print("\nAnalyse des mots-clés NON-SCA...")
    validated_non_sca = analyze_transcript(transcript, non_sca_keywords)

    save_validated_words(validated_sca, "mots_dits_sca.txt")
    save_validated_words(validated_non_sca, "mots_dits_non_sca.txt")

    print("\nMots-clés validés enregistrés dans les fichiers de sortie.")
