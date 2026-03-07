from .constant import *

def decode_output(probs, onset_threshold=0.5, frame_threshold=0.5, offset_threshold=0.5):
    onsets = (probs['onset'] > onset_threshold).cpu().numpy()
    frames = (probs['frame'] > frame_threshold).cpu().numpy()
    offsets = (probs['offset'] > offset_threshold).cpu().numpy()
    velocities = probs['velocity'].cpu().numpy()

    detected_notes = []

    for pitch in range(88):
        active_start = None
        
        for t in range(len(frames)):
            if onsets[t, pitch] and active_start is None:
                active_start = t
            
            elif active_start is not None:
                if not frames[t, pitch] or offsets[t, pitch]:
                    detected_notes.append({
                        'pitch': pitch + 21,
                        'start': active_start,
                        'end': t,
                        'velocity': velocities[active_start, pitch]
                    })
                    active_start = None
                    
                    if onsets[t, pitch]:
                        active_start = t
                
    return detected_notes