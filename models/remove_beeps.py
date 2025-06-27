import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from pydub import AudioSegment, effects
import os

# Prétraitement : normalisation + compression
def preprocess_audio(audio: AudioSegment) -> AudioSegment:
    # Compression (réduction des crêtes pour éviter les saturations)
    audio = effects.compress_dynamic_range(audio, threshold=-10.0, ratio=4.0)
    # Normalisation RMS (volume équilibré)
    return effects.normalize(audio)

# Traitement complet
def extract_speech(audio_path, output_path=None,
                   threshold=0.3, min_speech_ms=300,
                   speech_pad_ms=200, min_segment_ms=150):
    model = load_silero_vad()
    
    # 1. Charger + prétraiter
    audio = AudioSegment.from_wav(audio_path)
    preprocessed = preprocess_audio(audio)

    # 2. Sauvegarde temporaire
    tmp_path = audio_path.replace(".wav", "_tmp.wav")
    preprocessed.export(tmp_path, format="wav")

    # 3. Lecture pour Silero (numpy)
    wav = read_audio(tmp_path, sampling_rate=16000)

    # 4. Détection voix
    speech_timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False
    )

    # 5. Reconstruction audio
    cleaned = AudioSegment.empty()
    for seg in speech_timestamps:
        start_ms = seg['start'] * 1000 // 16000
        end_ms = seg['end'] * 1000 // 16000
        duration = end_ms - start_ms

        # On évite de garder les très courts bouts de voix
        if duration >= min_segment_ms:
            cleaned += preprocessed[start_ms:end_ms]

    # 6. Export
    out = output_path or audio_path.replace(".wav", "_cleaned.wav")
    cleaned.export(out, format="wav")
    
    # Nettoyage fichier temporaire
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("Audio nettoyé :", out)
    return out

if __name__ == "__main__":
    extract_speech("data/audio/hospital/4 184 504 - 3711612.wav")