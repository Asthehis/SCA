import whisperx
import gc
import torch

# on détermine les constantes importantes
device = "cuda"
audio_file = "data/raw/Audio-SCA-1.wav"
batch_size = 16
compute_type = "float16"

# chargement du modèle whisperX
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# transcription de l'audio
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
segments = result["segments"]
# print(segments)

# on supprime le modèle
gc.collect()
torch.cuda.empty_cache()
del model

# alignement des segments
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)

segments = result["segments"]
# print(segments)

# on supprime le modèle
gc.collect()
torch.cuda.empty_cache()
del model_a

# affichage du texte sous forme de dialogue avec les time code
print("\n==== Transcription ====\n")
output_txt = audio_file.replace(".wav", "_diarized.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments): # on parcourt les segments générés plus tôt
        # print(segment)
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip() # pour chaque segment on récupère le temps et le texte
        
        # on affiche le temps sous le format mm:ss
        def format_time(seconds):
            minutes = int(seconds // 60)
            sec = int(seconds % 60)
            return f"{minutes:02d}:{sec:02d}"

        print(f"[{format_time(start)} - {format_time(end)}] {text}")

        # Sauvegarde finale
        f.write(f"[{format_time(start)} - {format_time(end)}] {text}\n")

print(f" Transcription avec diarisation enregistrée dans : {output_txt}")
