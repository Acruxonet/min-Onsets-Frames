from .constant import *
mel_transform = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_mels=229,
    normalized=True
)

def load_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != SR:
        resampler = T.Resample(sr, SR)
        waveform = resampler(waveform)
    return waveform

amplitude_to_db = T.AmplitudeToDB(top_db=80)

def wav_to_mel(waveform):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_spec = mel_transform(waveform)

    mel_db = amplitude_to_db(mel_spec)

    return mel_db.squeeze(0).transpose(0, 1)


