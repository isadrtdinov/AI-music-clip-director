import os
import shutil
import yaml
import torch
import argparse
from clip_director import ClipDirector


parser = argparse.ArgumentParser(description="AI-music-clip-director")
parser.add_argument("--query", dest="query", required=True, type=str)
parser.add_argument("--style", dest="style", required=True, type=str)
parser.add_argument("--config", dest="config", required=False, default="config.yaml", type=str)
parser.add_argument("--ya_music_token", dest="ya_music_token", required=True, type=str)
parser.add_argument("--genius_token", dest="genius_token", required=True, type=str)
args = parser.parse_args()


def load_config(config_file):
    stream = open(config_file, 'r')
    config = yaml.safe_load(stream)
    return config


config = load_config(config_file=args.config)
song_file = config['song_file']
vocals_file = config['vocals_file']
video_file = config['video_file']
cache_dir = config['cache_dir']
cached_lyrics = config['cached_lyrics']
whisper_size = config['whisper_size']
whisper_sample_rate = config['whisper_sample_rate']
image_height = int(config['image_height'])
image_width = int(config['image_width'])
fps = int(config['fps'])
kandinsky_images_steps = int(config['kandinsky_images_steps'])
kandinsky_flow_steps = int(config['kandinsky_flow_steps'])
kandinsky_denoised_type = config['kandinsky_denoised_type']
kandinsky_dynamic_threshold_v = float(config['kandinsky_dynamic_threshold_v'])
kandinsky_sampler = config['kandinsky_sampler']
kandinsky_ddim_eta = float(config['kandinsky_ddim_eta'])
kandinsky_guidance_scale = float(config['kandinsky_guidance_scale'])
kandinsky_strength = float(config['kandinsky_strength'])
kandinsky_progress = bool(config['kandinsky_progress'])
kandinsky_prompt_perturbation = float(config['kandinsky_prompt_perturbation'])
with_img2img_transition = bool(config['with_img2img_transition'])


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_director = ClipDirector(device=device, whisper_size=whisper_size, whisper_beam_size=10, fps=fps,
                             image_height=image_height, image_width=image_width,
                             with_img2img_transition=with_img2img_transition,
                             kandinsky_images_steps=kandinsky_images_steps, kandinsky_flow_steps=kandinsky_flow_steps,
                             kandinsky_denoised_type=kandinsky_denoised_type,
                             kandinsky_dynamic_threshold_v=kandinsky_dynamic_threshold_v,
                             kandinsky_sampler=kandinsky_sampler, kandinsky_ddim_eta=kandinsky_ddim_eta,
                             kandinsky_guidance_scale=kandinsky_guidance_scale, kandinsky_strength=kandinsky_strength,
                             prompt_perturbation=kandinsky_prompt_perturbation, style=args.style)

lyrics, duration, language = clip_director.get_song_and_lyrics(
    args.query, song_file, args.ya_music_token, args.genius_token
)
if language == "Russian":
    clip_director.separate_vocals(song_file=song_file, out_file=vocals_file,
                                  whisper_sample_rate=whisper_sample_rate)
else:
    vocals_file = song_file

segments = clip_director.generate_alignment(song_file=vocals_file, lyrics_str=lyrics,
                                            language=language, duration=duration)
prompts, times = zip(*segments)
images_with_text = clip_director.generate_images(
    prompts=list(prompts), times=list(times), duration=duration
)

if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)

os.makedirs(cache_dir)
lyrics_desc = ''
for i, (image, text) in enumerate(images_with_text):
    filename = f'{i:05d}.jpg'
    path = os.path.join(cache_dir, filename)
    image.convert('RGB').save(path)
    lyrics_desc = filename + '\t' + text + '\n'

with open(cached_lyrics, 'w') as file:
    file.write(lyrics_desc)

shutil.make_archive(cache_dir, 'zip', cache_dir)
shutil.rmtree(cache_dir)

clip_director.create_video_clip(
    images_with_texts=images_with_text, song_file=song_file,
    video_file=video_file
)
