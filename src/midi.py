from constant import *

def load_midi(midi_path, total_frames, sr=16000, hop_length=512):

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    shape = (total_frames, 88)
    onset_label = np.zeros(shape, dtype=np.float32)
    offset_label = np.zeros(shape, dtype=np.float32)
    frame_label = np.zeros(shape, dtype=np.float32)
    velocity_label = np.zeros(shape, dtype=np.float32)

    sec_to_frame = sr / hop_length

    for instrument in midi_data.instruments:
        if instrument.is_drum or not (0 <= instrument.program <= 7):
            continue

        for note in instrument.notes:
            pitch = note.pitch - 21
            if not (0 <= pitch < 88):
                continue

            start_frame = int(round(note.start * sec_to_frame))
            end_frame = int(round(note.end * sec_to_frame))

            if start_frame >= total_frames:
                continue
            
            end_frame = min(max(end_frame, start_frame + 1), total_frames)


            onset_label[start_frame, pitch] = 1.0
            velocity_label[start_frame, pitch] = note.velocity / 127.0
            frame_label[start_frame:end_frame, pitch] = 1.0
            
            if end_frame < total_frames:
                offset_label[end_frame, pitch] = 1.0
                
    return {
        'onset': onset_label,
        'offset': offset_label,
        'frame': frame_label,
        'velocity': velocity_label
    }

def save_midi(path, notes):
    piano = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) 

    for n in notes:
        midi_note = pretty_midi.Note(
            velocity=int(n['velocity']),
            pitch=n['pitch'],
            start=n['start'],
            end=n['end']
        )
        instrument.notes.append(midi_note)

    piano.instruments.append(instrument)
    piano.write(path)