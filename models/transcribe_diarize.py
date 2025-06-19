import os
import json
import subprocess
import torch
import gc
from datetime import timedelta
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from huggingface_hub import login

def convert_m4a_to_wav(input_path, output_path, sample_rate=16000):
    if not os.path.exists(output_path):
        print(f" Conversion de {input_path} vers {output_path}...")
        try:
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-ar", str(sample_rate),
                "-ac", "1",
                output_path
            ], check=True)
            print(" Conversion réussie.")
        except subprocess.CalledProcessError:
            print(" Erreur lors de la conversion avec ffmpeg.")
    else:
        print(" Le fichier WAV existe déjà, pas de conversion nécessaire.")

def transcribe_and_diarize(audio_path):
    with open("secrets.json", "r") as f:
        hf_token = json.load(f)["use_auth_token"]
    
    login(token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(" Device:", device)

    # Conversion si nécessaire
    wav_path = audio_path.replace(".m4a", ".wav")
    convert_m4a_to_wav(audio_path, wav_path)

    # Transcription
    print(" Transcription...")
    model = WhisperModel("large-v2", device=device, compute_type="float16")
    segments, _ = model.transcribe(wav_path, beam_size=5)
    segment_list = [{
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip()
    } for seg in segments]
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Diarisation
    print(" Diarisation...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(torch.device(device))
    diarization = pipeline(wav_path)

    # Association speaker + texte
    print(" Fusion...")
    final_lines = []
    from pyannote.core import Segment

    def get_matching_speaker(seg_start, seg_end, diarization):
        seg = Segment(seg_start, seg_end)
        max_overlap = 0
        matched_speaker = "Unknown"

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap = seg & turn  # intersection entre les intervalles
            if overlap and overlap.duration > max_overlap:
                max_overlap = overlap.duration
                matched_speaker = speaker

        return matched_speaker

    for seg in segment_list:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"]

        speaker_label = get_matching_speaker(seg_start, seg_end, diarization)

        start_str = str(timedelta(seconds=int(seg_start)))
        end_str = str(timedelta(seconds=int(seg_end)))
        final_lines.append(f"[{start_str} - {end_str}] {speaker_label} : {seg_text}")


    # Sauvegarde finale
    output_txt = wav_path.replace(".wav", "_diarized.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(final_lines))
    print(f" Transcription avec diarisation enregistrée dans : {output_txt}")

if __name__ == "__main__":
    with open("data/last_filename.txt", "r", encoding="utf-8") as f:
        filename = f.read().strip()

transcribe_and_diarize(f"data/raw/{filename}")

