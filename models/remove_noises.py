import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from pydub import AudioSegment, effects
import os
from transcription import transcription

def preprocess_audio(audio: AudioSegment) -> AudioSegment:

    audio = effects.low_pass_filter(audio, 3500)
    audio = effects.high_pass_filter(audio, 120)
    audio = effects.compress_dynamic_range(audio, threshold=-17.0, ratio=5.0)
    return effects.normalize(audio, headroom=4.0)


# Traitement complet
def extract_speech(audio_path, output_path=None,
                   threshold=0.2, min_speech_ms=200,
                   speech_pad_ms=250, min_segment_ms=100, 
                   model=None):

    if model is None:
        model = load_silero_vad()

    audio = AudioSegment.from_wav(audio_path)
    preprocessed = preprocess_audio(audio)

    # Niveau RMS (root mean square) = volume perçu
    rms = preprocessed.rms

    # Ratio silence / son utile
    silence_threshold = preprocessed.dBFS - 16  # seuil pour considérer silence
    chunks = preprocessed[::100]  # 10 morceaux par seconde
    nb_silent_chunks = sum(1 for c in chunks if c.dBFS < silence_threshold)
    silence_ratio = nb_silent_chunks / len(chunks)
    rejeter = False

    if rms < 500:  # trop faible
        rejeter = True

    if silence_ratio > 0.8:  # que du blanc
        rejeter = True

    if rejeter == False:
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
    else:
        print("Audio rejeter")

if __name__ == "__main__":

    model = load_silero_vad()
    audio_dir = "data/audio/hospital"
    for file_name in os.listdir(audio_dir):
        if file_name.lower().endswith(".wav"):
            audio_path = os.path.join(audio_dir, file_name)
            print(f"\Traitement de : {audio_path}")
            cleaned_path = extract_speech(audio_path, model)
            if cleaned_path:
                transcription_text, rejected = transcription(cleaned_path)
            else:
                print("Fichier ignoré.")


