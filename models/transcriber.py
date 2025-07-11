import whisper
import torch
import gc
import os
import threading

PLOT_LOCK = threading.Lock()

class Transcriber:
    def __init__(self, audio_path, language="fr", verbose=True):
        self.audio_path = audio_path
        self.language = language
        self.device = "cuda"
        self.transcription = ""
        self.verbose = verbose

    def transcribe(self):

        # chargement du modèle whisperX
        model = whisper.load_model("large-v2", self.device)

        try:
            if self.verbose:
                print(f"Transcription de {self.audio_path} en cours...")
        # transcription de l'audio
            result = model.transcribe(self.audio_path, language=self.language, verbose=self.verbose)

            self.segments = result.get("segments", [])

            self.transcription = " ".join([seg["text"].strip() for seg in self.segments])
            if self.verbose:
                print(f"Transcription de {self.audio_path} terminée.")
        except Exception as e:
            print(f"Erreur lors de la transcription : {e}")
            raise

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            del model
        
    def save_transcript(self):
        if not self.segments:
            print("Aucune transcription à sauvegarder.")
            return
        
        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        output_path = os.path.join("data/transcript", f"{base_name}.txt")

        with PLOT_LOCK:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    for seg in self.segments:
                        start = self.format_time(seg["start"])
                        end = self.format_time(seg["end"])
                        f.write(f"[{start} - {end}] : {seg['text'].strip()}\n")

                print(f"Transcription sauvegardée : {output_path}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")
                raise

    @staticmethod
    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes:02d}:{sec:02d}"