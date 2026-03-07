from constant import * 

mel_transform = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_mels=229,
    normalized=True
)

def ToMel (audio_path):
    waveform, _ = torchaudio.load(audio_path)

    # Single Channel
    if (waveform.shape[0] > 1):
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_spec = mel_transform(waveform)

    mel_db = (mel_spec + 1e-10).log()

    mel_db = mel_db.squeeze(0).transpose(0, 1)

    return mel_db

def LoadMidi(midi_path, total_frames, sr=SR, hop_length=512):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    frame_label = np.zeros((total_frames, 88), dtype = np.float32)

    onset_label = np.zeros((total_frames, 88), dtype = np.float32)
    
    offset_label = np.zeros((total_frames, 88), dtype=np.float32)
    

    velocity_label = np.zeros((total_frames, 88), dtype = np.float32)

    sec_to_frame = sr / hop_length

    for instrument in midi_data.instruments:
        if instrument.is_drum or not (0 <= instrument.program <= 7):
            continue
        for note in instrument.notes:
            pitch = note.pitch - 21
            if (0 <= pitch < 88):
                start_frame = int(round(note.start * sec_to_frame))

                end_frame = int(round(note.end * sec_to_frame))

                if (end_frame == start_frame):
                    end_frame = start_frame + 1

                if (start_frame < total_frames):
                    frame_label[start_frame:min(end_frame, total_frames), pitch] = 1.0
                    onset_label[start_frame, pitch] = 1.0
                    velocity_label[start_frame, pitch] = note.velocity / 127.0
                if end_frame < total_frames:
                        offset_label[end_frame, pitch] = 1.0                
    return frame_label, onset_label, velocity_label

def GetChunk(mel, frame, onset, velocity, window_size = 256):
    start = np.random.randint(0, len(mel) - window_size)
    return (mel[start:start+window_size], onset[start:start+window_size], 
            frame[start:start+window_size], velocity[start:start+window_size])