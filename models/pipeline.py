from audio_processor import AudioProcessor
from transcriber import Transcriber
from file_cleaner import FileCleaner
import os

def clean():
    # Nettoyage des audio
    input_dir = "data/audio/hospital"
    cleaner = FileCleaner(input_dir)
    cleaner.remove_cleaned_files()         # Supprime tous les fichiers *_cleaned.wav

    # Nettoyage des transcriptions
    input_dir = "data/transcript"
    cleaner = FileCleaner(input_dir)
    cleaner.remove_cleaned_files()         # Supprime tous les fichiers *_cleaned.txt


def process_audio_pipeline(audio_path):
    print(f"\n--- Traitement de {audio_path} ---")

    # Prétraitement + VAD
    processor = AudioProcessor(audio_path)
    if not processor.process():
        print("Audio rejeté (qualité)")
        return

    # Transcription
    transcriber = Transcriber(processor.cleaned_path)
    transcriber.transcribe()
    transcriber.save_transcript()
    print("Audio traité avec succès.")

if __name__ == "__main__":
    clean()
    input_dir = "data/audio/hospital"
    for fname in os.listdir(input_dir):
        if fname.endswith(".wav"):
            process_audio_pipeline(os.path.join(input_dir, fname))
