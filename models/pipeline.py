import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from audio_processor import AudioProcessor
from transcriber import Transcriber
from file_cleaner import FileCleaner

from threading import current_thread

NUM_THREADS = 4
INPUT_DIR = "data/audio/hospital"
TRANSCRIPT_DIR = "data/transcript"

def clean():
    # Nettoyage des audio & transcriptions
    for directory in [INPUT_DIR, TRANSCRIPT_DIR]:
        cleaner = FileCleaner(directory)
        cleaner.remove_cleaned_files()

def process_audio_pipeline(audio_path):
    try:
        thread_name = current_thread().name
        print(f"\n[{thread_name}] : Traitement de {audio_path}")

        processor = AudioProcessor(audio_path)
        if not processor.process():
            print(f"[{thread_name}] : Audio rejeté (qualité) pour {audio_path}")
            return

        # Transcription
        transcriber = Transcriber(processor.cleaned_path)
        transcriber.transcribe()
        transcriber.save_transcript()

        print(f"[{thread_name}] : Audio traité avec succès ({audio_path}).")

    except Exception as e:
        print(f"[{thread_name}] : Erreur lors du traitement de ({audio_path}) : {e}")

def main():
    clean()

    audio_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".wav") and not f.endswith("_cleaned.wav")
    ]

    if not audio_files:
        print("Aucun fichier audio à traiter.")
        return
    
    print(f"Lancement du traitement avec {NUM_THREADS} threads...\n")
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_audio_pipeline, audio) for audio in audio_files]

        for future in as_completed(futures):
            future.result()

    print("\nTraitement terminé pour tous les fichiers.")

if __name__ == "__main__":
    main()