import re
import json

# Charger les bases de données de keywords
with open("data/sca_words.json", "r", encoding="utf-8") as f:
    sca_data = json.load(f)

with open("data/non_sca_words.json", "r", encoding="utf-8") as f:
    non_sca_data = json.load(f)

# Créer les dictionnaires de sévérité
symptomes_severite = {}
for item in sca_data["keywords"]:
    symptomes_severite[item["word"]] = item["severity"]
    for synonym in item["synonyms"]:
        symptomes_severite[synonym] = item["severity"]

non_sca_severite = {}
for item in non_sca_data["keywords"]:
    non_sca_severite[item["word"]] = item["severity"]
    for synonym in item["synonyms"]:
        non_sca_severite[synonym] = item["severity"]

# 📂 Lire les mots détectés depuis les deux fichiers de sortie
def read_detected_words(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        print(f" Fichier manquant : {filepath}")
        return []


mots_sca = read_detected_words("mots_dits_sca.txt")
mots_non_sca = read_detected_words("mots_dits_non_sca.txt")



# Simuler un texte complet pour l'extraction d'âge/genre
# Lire directement le texte complet depuis le fichier original si possible
with open("data/last_filename.txt", "r", encoding="utf-8") as f:
    filename = f.read().strip()

diarized_path = f"data/raw/{filename.replace('.m4a', '_diarized.txt')}"
with open(diarized_path, "r", encoding="utf-8") as f:
    texte = f.read().lower()


#  Extraction de l'âge du patient
match_age = re.search(r"(\d{2})\s*ans", texte)
age = int(match_age.group(1)) if match_age else None

#  Détection du genre
homme = bool(re.search(r"\b(il|homme|monsieur)\b", texte))
femme = bool(re.search(r"\b(elle|femme|madame)\b", texte))

#  Vérification de l'âge critique
age_critique = (homme and age and age >= 50) or (femme and age and age >= 55)

#  Symptômes confirmés
symptomes_detectes = [s for s in symptomes_severite if any(s in ligne for ligne in mots_sca)]
symptomes_non_sca = [s for s in non_sca_severite if any(s in ligne for ligne in mots_non_sca)]

#  Calcul du score
score_total = sum(symptomes_severite[s] for s in symptomes_detectes) - sum(non_sca_severite[s] for s in symptomes_non_sca)

#  Bonus si âge critique
if age_critique:
    score_total += 5

score_total = max(score_total, 0)

#  Résultats
print("Âge détecté :", age)
print("Genre détecté :", "Homme" if homme else "Femme" if femme else "Inconnu")
print("Symptômes reconnus :", symptomes_detectes)
print("Symptômes non-SCA reconnus :", symptomes_non_sca)
print(" Score total de sévérité ajusté :", score_total)


# Ajout à la fin de score.py
with open("data/score_final.txt", "w", encoding="utf-8") as f:
    f.write(str(score_total))
