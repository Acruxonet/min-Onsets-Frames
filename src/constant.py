import torch
import random
import torchaudio
import numpy as np
import pretty_midi
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
SR = 16000
