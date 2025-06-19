import subprocess

# Étape 1 : Transcription + Diarisation
print("=== Étape 1 : Transcription & Diarisation ===")
subprocess.run(["python", "models/transcribe_diarize.py"], check=True)

# Étape 2 : Compréhension des symptômes (affirmation/négation)
print("\n=== Étape 2 : Compréhension (affirmation des symptômes) ===")
subprocess.run(["python", "models/comprehension.py"], check=True)

# Étape 3 : Calcul du score
print("\n=== Étape 3 : Calcul du score final ===")
subprocess.run(["python", "models/score.py"], check=True)
