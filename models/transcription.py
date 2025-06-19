import whisperx
import gc
import torch

device = "cuda"
audio_file = "data/raw/Audio-SCA-1.wav"
batch_size = 16
compute_type = "float16"

# Chargement du modèle whisperX
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# Transcription de l'audio
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
segments = result["segments"]
print(segments)

# On supprime le modèle
gc.collect()
torch.cuda.empty_cache()
del model

# Align segments
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)

segments = result["segments"]
print(segments)

# On supprime le modèle
gc.collect()
torch.cuda.empty_cache()
del model_a

# Format as dialogue-style transcript with timestamps
print("\n==== Transcription ====\n")
for i, segment in enumerate(segments):
    start = segment["start"]
    end = segment["end"]
    text = segment["text"].strip()
    
    # Format time as MM:SS
    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        return f"{minutes:02d}:{sec:02d}"

    print(f"[{format_time(start)} - {format_time(end)}] {text}")
