import subprocess
import sys


from get_song import get_lyrics
from align_segments import align_segments

import io
import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
import urllib
import tarfile
import whisper
import torchaudio

from scipy.io import wavfile
from tqdm.notebook import tqdm

song_file = "song.mp3"
lyrics_file = "ans.txt"
query = "Never"
transcription_pickle_file = "transcription.pickle"


model = whisper.load_model("medium")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)


lyrics, language = get_lyrics(query, song_file)

with open(lyrics_file, 'w') as file:
    file.write(lyrics)

options = dict(language=language, beam_size=10, best_of=10)
transcribe_options = dict(task="transcribe", **options)

transcription = model.transcribe(song_file, **transcribe_options)



pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



ans = align_segments(lyrics_file, transcription_pickle_file)
for i in range(len(ans)):
    print(ans[i]["text"], ans[i]["start"], ans[i]["end"])
