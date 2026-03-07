from constant import * 
from mel import *
from midi import *




def GetChunk(mel, frame, onset, velocity, window_size = 256):
    start = np.random.randint(0, len(mel) - window_size)
    return (mel[start:start+window_size], onset[start:start+window_size], 
            frame[start:start+window_size], velocity[start:start+window_size])