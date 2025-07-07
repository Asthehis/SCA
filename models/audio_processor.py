from pydub import AudioSegment, effects
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import iirnotch, lfilter

MIN_RMS = 1400
MAX_SATURATION_FRAMES = 3 

class AudioProcessor:
    def __init__(self, audio_path, verbose=True):
        self.audio_path = audio_path
        self.cleaned_path = audio_path.replace(".wav", "_cleaned.wav")
        self.original_audio = AudioSegment.from_wav(audio_path)
        self.preprocessed_audio = None
        self.val_model = load_silero_vad()
        self.should_reject = False
        self.rms = 0
        self.saturation_count = 0
        self.duration_sec = len(self.original_audio) / 1000
        self.dominant_freq = 0
        self.mean_freq = 0
        self.bandwidth = 0
        self.verbose = verbose

    def preprocess(self):
        audio = self.original_audio
        audio = effects.low_pass_filter(audio, 1300)
        audio = effects.high_pass_filter(audio, 150)
        audio = effects.compress_dynamic_range(audio, threshold=-16.0, ratio=4.0)
        self.preprocessed_audio = effects.normalize(audio, headroom=4.0)

    def analyze_quality(self):
        audio = self.preprocessed_audio
        self.rms = audio.rms

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)

        max_sample = max(abs(s) for s in samples)
        self.saturation_count = sum(1 for s in samples if abs(s) >= max_sample * 0.98)

        self.analyze_frequency(samples, audio.frame_rate)

        if self.rms < MIN_RMS or self.saturation_count > MAX_SATURATION_FRAMES:
            self.should_reject = True
        
        if self.verbose:
            print(f"Durée: {self.duration_sec:.2f}s")
            print(f"RMS: {self.rms}")
            print(f"Saturation frames: {self.saturation_count}")
            print(f"Dominant freq: {self.dominant_freq:.1f} Hz")
            print(f"Mean freq: {self.mean_freq:.1f} Hz")
            print(f"Bandwidth: {self.bandwidth:.1f} Hz")

    def analyze_frequency(self, samples, sr):
        samples = samples - np.mean(samples)
        n = len(samples)
        yf = np.abs(rfft(samples))/ n
        xf = rfftfreq(n, 1 / sr)

        self.dominant_freq = xf[np.argmax(yf)]
        self.mean_freq = np.sum(xf * yf) / np.sum(yf)
        self.bandwidth = np.sqrt(np.sum(((xf - self.mean_freq) ** 2) * yf) / np.sum(yf))

        self.save_plot(samples, sr, xf, yf)

    def save_plot(self, samples, sr, xf, yf):
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        t = np.arange(len(samples)) / sr
        plt.plot(t, samples, color='gray')
        plt.title(f"{base} - Time Domain")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        plt.plot(xf, yf, color='blue')
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{base}.png")
        plt.close()

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
                    "dominant_freq", "mean_freq", "bandwidth", "rejected"
                ])

            writer.writerow([
                os.path.basename(self.audio_path),
                round(self.duration_sec, 2),
                self.rms,
                self.saturation_count,
                round(self.dominant_freq, 2),
                round(self.mean_freq, 2),
                round(self.bandwidth, 2),
                self.should_reject
            ])

    def process(self):
        self.preprocess()
        self.analyze_quality()
        if not self.should_reject:
            self.apply_vad()
        self.log_to_csv()
        return not self.should_reject