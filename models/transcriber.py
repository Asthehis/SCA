import whisperx
import torch
import gc
import os

class Transcriber:
    def __init__(self, audio_path, language="fr", verbose=True):
        self.audio_path = audio_path
        self.language = language
        self.device = "cuda"
        self.batch_size = 4
        self.compute_type = "float16"
        self.transcription = ""
        self.verbose = verbose

    def transcribe(self):

        # chargement du modèle whisperX
        model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type, language=self.language)

        # transcription de l'audio
        audio = whisperx.load_audio(self.audio_path)
        result = model.transcribe(audio, batch_size=self.batch_size)
        segments = result["segments"]
        # print(segments)

        # on supprime le modèle
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        del model

        # alignement des segments
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(segments, model_a, metadata, audio, self.device, return_char_alignments=False)

        # on supprime le modèle
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        del model_a

        self.segments = result["segments"]
        self.transcription = " ".join([seg["text"].strip() for seg in self.segments])
    
    def save_transcript(self):
        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        output_path = os.path.join("data/transcript", f"{base_name}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in self.segments:
                start = self.format_time(seg["start"])
                end = self.format_time(seg["end"])
                f.write(f"[{start} - {end}] : {seg['text'].strip()}\n")

        print(f"Transcription sauvegardée : {output_path}")

    @staticmethod
    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes:02d}:{sec:02d}"