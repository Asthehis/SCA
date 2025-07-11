import os
import csv
import threading
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor, NoiseGate, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment, effects
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

MIN_RMS = 2000
MAX_RMS = 4000  
MAX_SATURATION_FRAMES = 30
MAX_NOISE_LEVEL = 0.25
MIN_SPECTRAL_CENTROID = 600 
MAX_SPECTRAL_CENTROID = 1200

PLOT_LOCK = threading.Lock()

class AudioProcessor:
    def __init__(self, audio_path, verbose=True):
        self.audio_path = audio_path
        self.cleaned_path = audio_path.replace(".wav", "_cleaned.wav")
        self.original_audio = AudioSegment.from_wav(audio_path)
        self.preprocessed_audio = None
        self.val_model = load_silero_vad()
        self.should_reject = False
        self.rejection_reasons = []

        # metrics
        self.rms = 0
        self.saturation_count = 0
        self.duration_sec = len(self.original_audio) / 1000
        self.dominant_freq = 0
        self.mean_freq = 0
        self.bandwidth = 0
        self.noise_level = 0
        self.spectral_centroid = 0
        self.spectral_rolloff = 0
        self.zero_crossing_rate = 0

        self.verbose = verbose
        self.enhanced_samples = None

        self.plot_id = str(uuid.uuid4())[:8]

    def preprocess(self):
        with AudioFile(self.audio_path) as f:
            audio = f.read(f.frames) 
            sr = f.samplerate

            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=100),
                LowpassFilter(cutoff_frequency_hz=1300),
                Compressor(threshold_db=-20, ratio=3.0),
                NoiseGate(threshold_db=-45, ratio=3.0),
                Reverb(room_size=0.1, damping=0.8, wet_level=0.05, dry_level=0.95),
            ])
            effected = board(audio, sample_rate=sr)

        # Convertir les données audio en AudioSegment directement
        # effected est un array numpy, on le convertit en bytes pour pydub
        if len(effected.shape) == 1:
            effected = effected.reshape(1, -1)
        
        # Convertir en int16 pour pydub, clip pour éviter la saturation
        effected_clipped = np.clip(effected * 32767, -32767, 32767).astype(np.int16)
        
        # Créer un AudioSegment à partir des données
        audio_segment = AudioSegment(
            effected_clipped.tobytes(),
            frame_rate=int(sr),
            sample_width=2,  # 16-bit = 2 bytes
            channels=1
        )
        
        self.preprocessed_audio = effects.normalize(audio_segment, headroom=6.0)

    def analyze_quality(self):
        audio = self.preprocessed_audio
        self.rms = audio.rms

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        if len(samples) > 0:
            samples = samples / np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else samples

        max_sample = np.max(np.abs(samples))
        self.saturation_count = np.sum(np.abs(samples) >= max_sample * 0.95)

        self.analyze_speech_quality(samples, audio.frame_rate)
        self.analyze_frequency(samples, audio.frame_rate)

        self.check_rejection_criteria()

    def analyze_speech_quality(self, samples, sr):
        """
        
        """
        # on va essayer d'estimer le bruit dans les premières et dernières ms (~500)
        zero_crossings = np.sum(np.diff(np.signbit(samples)))
        self.zero_crossing_rate = zero_crossings / len(samples)

        # on va ensuite estimer le niveau de bruit global de l'audio
        self.estimate_noise_level(samples, sr)

    def estimate_noise_level(self, samples, sr):
        """
        
        """
        try:
            tmp_path = self.audio_path.replace(".wav", "_noise_tmp.wav")
            self.preprocessed_audio.export(tmp_path, format="wav")

            wav = read_audio(tmp_path, sampling_rate=16000)
            speech_segments = get_speech_timestamps(
                wav, self.val_model, sampling_rate=16000,
                threshold=0.3, min_speech_duration_ms=100
            )

            os.remove(tmp_path)

            if speech_segments:
                speech_mask = np.zeros(len(samples), dtype=bool)
                for seg in speech_segments:
                    start_idx = int(seg['start'] * sr / 16000)
                    end_idx = int(seg['end'] * sr / 16000)
                    start_idx = max(0, min(start_idx, len(samples)))
                    end_idx = max(0, min(end_idx, len(samples)))
                    speech_mask[start_idx:end_idx] = True

                noise_samples = samples[~speech_mask]
                if len(noise_samples) > 0:
                    self.noise_level = np.std(noise_samples)
                else:
                    self.noise_level = 0
            else:
                self.noise_level = np.std(samples)
            
        except Exception as e:
            if self.verbose:
                print(f"Erreur lors de l'estimation du bruit: {e}")
            self.noise_level = 1.0

    def analyze_frequency(self, samples, sr):
        samples = samples - np.mean(samples)
        n = len(samples)
        yf = np.abs(rfft(samples))/ n
        xf = rfftfreq(n, 1 / sr)

        if len(yf) > 0:
            self.dominant_freq = xf[np.argmax(yf)]
            self.mean_freq = np.sum(xf * yf) / np.sum(yf) if np.sum(yf) > 0 else 0
            self.bandwidth = np.sqrt(np.sum(((xf - self.mean_freq) ** 2) * yf) / np.sum(yf)) if np.sum(yf) > 0 else 0

            self.spectral_centroid = np.sum(xf * yf) / np.sum(yf) if np.sum(yf) > 0 else 0

            cumulative_energy = np.cumsum(yf)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            self.spectral_rolloff = xf[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            self.dominant_freq = 0
            self.mean_freq = 0
            self.bandwidth = 0
            self.spectral_centroid = 0
            self.spectral_rolloff = 0

        self.save_plot(samples, sr, xf, yf)

    def check_rejection_criteria(self):
        """
        """
        self.should_reject = False
        self.rejection_reasons = []

        if self.rms < MIN_RMS:
            self.should_reject = True
            self.rejection_reasons.append(f"RMS trop faible: {self.rms}")
        elif self.rms > MAX_RMS:
            self.should_reject = True
            self.rejection_reasons.append(f"RMS trop élevé: {self.rms}")

        if self.saturation_count > MAX_SATURATION_FRAMES:
            self.should_reject = True
            self.rejection_reasons.append(f"Saturation excessive: {self.saturation_count}")

        if self.noise_level > MAX_NOISE_LEVEL:
            self.should_reject = True
            self.rejection_reasons.append(f"Bruit excessif: {self.noise_level:.2f}")

        if not (MIN_SPECTRAL_CENTROID <= self.spectral_centroid <= MAX_SPECTRAL_CENTROID):
            self.should_reject = True
            self.rejection_reasons.append(f"Centroïde spectral hors limites: {self.spectral_centroid:.0f}Hz")

        if self.verbose and self.should_reject:
            print(f"Audio rejeté - Raisons: {', '.join(self.rejection_reasons)}")

    def save_plot(self, samples, sr, xf, yf):
        base = os.path.splitext(os.path.basename(self.audio_path))[0]

        with PLOT_LOCK:
            fig = plt.figure(figsize=(12, 8))

            try:
                # Signal temporel
                plt.subplot(2, 2, 1)
                t = np.arange(len(samples)) / sr
                plt.plot(t, samples, color='gray', alpha=0.7)
                plt.title(f"{base} - Signal temporel")
                plt.xlabel("Temps (s)")
                plt.ylabel("Amplitude")

                # Spectre fréquentiel
                plt.subplot(2, 2, 2)
                plt.plot(xf, yf, color='blue')
                plt.axvline(self.spectral_centroid, color='red', linestyle='--', label=f'Centroïde: {self.spectral_centroid:.0f}Hz')
                plt.title("Spectre fréquentiel")
                plt.xlabel("Fréquence (Hz)")
                plt.ylabel("Magnitude")
                plt.legend()

                # Métriques de qualité
                plt.subplot(2, 2, 3)
                metrics = [
                    f"RMS: {self.rms}",
                    f"Bruit: {self.noise_level:.2f}",
                    f"Centroïde: {self.spectral_centroid:.0f}Hz"
                ]
                for i, metric in enumerate(metrics):
                    plt.text(0.1, 0.9 - i*0.15, metric, transform=plt.gca().transAxes)
                plt.title("Métriques de qualité")
                plt.axis('off')

                plt.tight_layout()
                os.makedirs("plots", exist_ok=True)
                temp_path = f"plots/{base}_{self.plot_id}.png"
                final_path = f"plots/{base}.png"
                
                plt.savefig(temp_path, dpi=150, bbox_inches='tight')
                
                # Renommer le fichier temporaire vers le nom final
                if os.path.exists(temp_path):
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    os.rename(temp_path, final_path)
                
            finally:
                # Nettoyer la figure pour éviter les fuites mémoire
                plt.close(fig)
                plt.clf()  # Clear current figure
                plt.cla()  # Clear current axes

    def apply_vad(self):
        tmp_path = self.audio_path.replace(".wav", "_tmp.wav")
        self.preprocessed_audio.export(tmp_path, format="wav")
        wav = read_audio(tmp_path, sampling_rate=16000)
        speech_segments = get_speech_timestamps(
            wav, self.val_model, sampling_rate=16000,
            threshold=0.2, min_speech_duration_ms=150,
            speech_pad_ms=300
        )
        os.remove(tmp_path)

        cleaned = AudioSegment.empty()
        for seg in speech_segments:
            start_ms = seg['start'] * 1000 // 16000
            end_ms = seg['end'] * 1000 // 16000
            if (end_ms - start_ms) >= 100:
                cleaned += self.preprocessed_audio[start_ms:end_ms]
            
        cleaned.export(self.cleaned_path, format="wav")

    def log_to_csv(self, output_csv="audio_quality_log.csv"):
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "file", "duration", "rms", "saturation_count", 
                    "dominant_freq", "mean_freq", "bandwidth", 
                    "noise_level", "spectral_centroid", 
                    "spectral_rolloff", "zero_crossing_rate",
                    "rejected", "rejection_reasons"
                ])

            writer.writerow([
                os.path.basename(self.audio_path),
                round(self.duration_sec, 2),
                self.rms,
                self.saturation_count,
                round(self.dominant_freq, 2),
                round(self.mean_freq, 2),
                round(self.bandwidth, 2),
                round(self.noise_level, 3),
                round(self.spectral_centroid, 2),
                round(self.spectral_rolloff, 2),
                round(self.zero_crossing_rate, 4),
                self.should_reject,
                "; ".join(self.rejection_reasons) if self.rejection_reasons else ""
            ])

    def process(self):
        self.preprocess()
        self.analyze_quality()
        if not self.should_reject:
            self.apply_vad()
        self.log_to_csv()
        return not self.should_reject