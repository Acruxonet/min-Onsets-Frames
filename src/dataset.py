from .mel import wav_to_mel, SR
from .midi import load_midi
from .constant import *

class PianoDataset(Dataset):
    def __init__(self, data_list, segment_seconds=20.0, hop_length=512):
        self.data_list = data_list
        self.segment_samples = int(segment_seconds * SR)
        self.total_frames = self.segment_samples // hop_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, midi_path = self.data_list[index]
        total_samples = torchaudio.info(audio_path).num_frames
        start_sample = random.randint(0, max(0, total_samples - self.segment_samples))

        waveform, _ = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=self.segment_samples)
        
        if waveform.shape[1] < self.segment_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_samples - waveform.shape[1]))

        mel = wav_to_mel(waveform)
        labels = load_midi(midi_path, self.total_frames, offset_time=start_sample / SR)

        return {'mel': mel, **labels}
