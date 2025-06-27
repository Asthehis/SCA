from pydub import AudioSegment
import torchaudio
import os

def toWav(audio_path: str) -> str:
    """
    Convertit un fichier .mp3 en .wav (16kHz mono).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Fichier introuvable: {audio_path}.")
    
    base, ext = os.path.splitext(audio_path)
    wav_path = base + ".wav"

    audio = AudioSegment.from_file(audio_path, format=ext[1:])
    audio = audio.set_frame_rate(16000).set_channels(1)

    audio.export(wav_path, format="wav")
    return wav_path

# script principal
if __name__ == "__main__":
    audio_dir = "data/audio/23032024_juin"
    for file_name in os.listdir(audio_dir):
        if file_name.lower().endswith(".mp3"):
            audio_path = os.path.join(audio_dir, file_name)
            toWav(audio_path)