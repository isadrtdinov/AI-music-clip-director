import argparse

parser = argparse.ArgumentParser(description="AI-music-clip-director")
parser.add_argument("--query", dest="query", required=True, type=str)
parser.add_argument("--style", dest="style", required=True, type=str)
parser.add_argument("--config", dest="config", required=False, default="config.yaml", type=str)
parser.add_argument("--ya_music_token", dest="ya_music_token", required=True, type=str)
parser.add_argument("--genius_token", dest="genius_token", required=True, type=str)

args = parser.parse_args()

from composer import ClipDirector

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
import whisper
from scipy.io import wavfile
from tqdm.notebook import tqdm
import yaml


def load_config(config_file):
    stream = open(config_file, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary


dictionary = load_config(config_file=args.config)
out_sample_rate = int(dictionary['out_sample_rate'])
song_file = dictionary['song_file']
vocals_file = dictionary['vocals_file']
whisper_size = dictionary['whisper_size']
image_height = int(dictionary['image_height'])
image_width = int(dictionary['image_width'])
fps = int(dictionary['fps'])
kandinsky_images_steps = int(dictionary['kandinsky_images_steps'])
kandinsky_flow_steps = int(dictionary['kandinsky_flow_steps'])
kandinsky_denoised_type = dictionary['kandinsky_denoised_type']
kandinsky_dynamic_threshold_v = float(dictionary['kandinsky_dynamic_threshold_v'])
kandinsky_sampler = dictionary['kandinsky_sampler']
kandinsky_ddim_eta = float(dictionary['kandinsky_ddim_eta'])
kandinsky_guidance_scale = float(dictionary['kandinsky_guidance_scale'])
kandinsky_strength = float(dictionary['kandinsky_strength'])
kandinsky_progress = bool(dictionary['kandinsky_progress'])
kandinsky_prompt_perturbation = float(dictionary['prompt_perturbation'])
with_img2img_transition = bool(dictionary['with_img2img_transition'])
style = dictionary['style']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_director = ClipDirector(device=DEVICE, whisper_size=whisper_size, whisper_beam_size=10, fps=fps,
                             image_height=image_height, image_width=image_width,
                             with_img2img_transition=with_img2img_transition,
                             kandinsky_images_steps=kandinsky_images_steps, kandinsky_flow_steps=kandinsky_flow_steps,
                             kandinsky_denoised_type=kandinsky_denoised_type,
                             kandinsky_dynamic_threshold_v=kandinsky_dynamic_threshold_v,
                             kandinsky_sampler=kandinsky_sampler, kandinsky_ddim_eta=kandinsky_ddim_eta,
                             kandinsky_guidance_scale=kandinsky_guidance_scale, kandinsky_strength=kandinsky_strength,
                             prompt_perturbation=kandinsky_prompt_perturbation, style=style)

lyrics, title, artist, duration, language = clip_director.get_song_and_lyrics(args.query, song_file, args.ya_music_token, args.genius_token)
if language == "Russian":
    clip_director.separate_vocals(song_file=song_file, out_file=vocals_file, out_sample_rate=out_sample_rate)
else:
    vocals_file = song_file

segments = clip_director.generate_alignment(song_file=vocals_file, lyrics_str=lyrics, language=language)
prompts = []
times = []
for i in segments:
    prompts.append(i[0])
    times.append(i[1])
all_images = clip_director.generate_images(prompts=prompts, times=times, title=title, artist=artist, duration=duration)
video_file = clip_director.create_video_clip(images_with_texts=all_images, song_file=song_file

