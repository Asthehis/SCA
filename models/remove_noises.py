import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from pydub import AudioSegment, effects
import os
import json

def preprocess_audio(audio: AudioSegment, profile: str = "balanced") -> AudioSegment:
    if profile == "balanced":
        # Compromis : compression + normalisation douce
        audio = effects.compress_dynamic_range(audio, threshold=-15.0, ratio=2.5)
        return effects.normalize(audio, headroom=4.0)  # ↓ pour éviter de "pousser" trop fort

    elif profile == "very_noisy":
        audio = effects.low_pass_filter(audio, 3500)
        audio = effects.high_pass_filter(audio, 120)
        audio = effects.compress_dynamic_range(audio, threshold=-25.0, ratio=6.0)
        return effects.normalize(audio, headroom=3.0)

    elif profile == "saturation_only":
        audio = effects.compress_dynamic_range(audio, threshold=-12.0, ratio=6.5)
        return effects.normalize(audio, headroom=5.0)

    elif profile == "echo_or_external":
        audio = audio.low_pass_filter(3300)
        audio = audio.high_pass_filter(150)
        audio = effects.compress_dynamic_range(audio, threshold=-20.0, ratio=5.0)
        return effects.normalize(audio, headroom=4.0)

    elif profile == "pompier_call":
        return effects.normalize(audio, headroom=6.0)

    return effects.normalize(audio, headroom=4.0)

# Traitement complet
def extract_speech(audio_path, output_path=None, profile="balanced",
                   threshold=0.2, min_speech_ms=200,
                   speech_pad_ms=250, min_segment_ms=100, 
                   model=None):

    if model is None:
        model = load_silero_vad()

    audio = AudioSegment.from_wav(audio_path)
    preprocessed = preprocess_audio(audio, profile=profile)

    # Sauvegarde temporaire
    tmp_path = audio_path.replace(".wav", "_tmp.wav")
    preprocessed.export(tmp_path, format="wav")

    # Lecture pour Silero (numpy)
    wav = read_audio(tmp_path, sampling_rate=16000)

    # Détection voix
    speech_timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False
    )

    # Reconstruction audio
    cleaned = AudioSegment.empty()
    for seg in speech_timestamps:
        start_ms = seg['start'] * 1000 // 16000
        end_ms = seg['end'] * 1000 // 16000
        duration = end_ms - start_ms

        # On évite de garder les très courts bouts de voix
        if duration >= min_segment_ms:
            cleaned += preprocessed[start_ms:end_ms]

    # Export
    out = output_path or audio_path.replace(".wav", "_cleaned.wav")
    # if os.path.exists(out):
    #     print(f"Déjà traité : {out}")
    #     return out

    cleaned.export(out, format="wav")
    
    # Nettoyage fichier temporaire
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("Audio nettoyé :", out)
    return out

if __name__ == "__main__":

    with open("models/audio_profiles.json", "r") as f:
        audio_profiles = json.load(f)

    model = load_silero_vad()

    for entry in audio_profiles:
        path = f"data/audio/hospital/{entry['filename']}"
        if not os.path.exists(path):
            print(f"[AVERTISSEMENT] Fichier manquant : {path}")
            continue

        profile = entry["profile"]
        print(f"Traitement de {entry['filename']} avec profil : {profile}")
        extract_speech(path, profile=profile, model=model)

