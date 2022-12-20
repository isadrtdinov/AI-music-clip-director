import argparse

parser = argparse.ArgumentParser(description="AI-music-clip-director")
parser.add_argument("--query", dest="query", required=True)
parser.add_argument("--style", dest="style", required=True)
parser.add_argument("--config", dest="config", required=True)
parser.add_argument("--ya_music_token", dest="ya_music_token", required=True)
parser.add_argument("--genius_token", dest="genius_token", required=True)

args = parser.parse_args()

from composer import ClipDirector
from separate_vocals import separate_vocals

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
import yaml


def load_config():
    stream = open("config.yaml", 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary


dictionary = load_config()
out_sample_rate = int(dictionary['out_sample_rate'])
song_file = dictionary['song_file']
vocals_file = dictionary['vocals_file']
whisper_size = dictionary['whisper_size']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_director = ClipDirector(device=DEVICE, whisper_size=whisper_size, whisper_beam_size=10, fps=20)

lyrics, language = clip_director.get_song_and_lyrics(args.query, song_file, args.ya_music_token, args.genius_token)
clip_director.separae_vocals(song_file=song_file, out_file=vocals_file, out_sample_rate=out_sample_rate)

segments = clip_director.generate_alignment(song_file=vocals_file, lyrics_str=lyrics, language=language)
prompts = []
for i in segments:
    prompts.append(i[0])
all_images = clip_director.generate_images(prompts=prompts)
