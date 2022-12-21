import re
import string
import textwrap
import whisper
import numpy as np
from copy import deepcopy
from kandinsky2 import get_kandinsky2
from align_segments import align_segments
from get_song import get_lyrics
from separate_vocals import separate_vocals
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip, CompositeAudioClip, ImageClip,
    VideoFileClip, concatenate_videoclips,
)


class ClipDirector(object):
    def __init__(self, device,
                 whisper_size: str = 'medium', whisper_beam_size: int = 10,
                 fps: int = 20, image_height: int = 512, image_width: int = 512, with_img2img_transition: bool = True,
                 kandinsky_images_steps: int = 75, kandinsky_flow_steps: int = 100,
                 kandinsky_denoised_type: str = "dynamic_threshold", kandinsky_dynamic_threshold_v: float = 99.5,
                 kandinsky_sampler: str = "kandinsky_sampler", kandinsky_ddim_eta: float = 0.05,
                 kandinsky_guidance_scale: float = 7, kandinsky_strength: float = 0.9, kandinsky_progress: bool = True,
                 prompt_perturbation: float = 0.25, style: str = ""):
        self.whisper_size = whisper_size
        self.whisper_beam_size = whisper_beam_size
        self.fps = fps
        self.image_height = image_height
        self.image_width = image_width
        self.with_img2img_transition = with_img2img_transition
        self.kandinsky_images_steps = kandinsky_images_steps
        self.kandinsky_flow_steps = kandinsky_flow_steps
        self.kandinsky_denoised_type = kandinsky_denoised_type
        self.kandinsky_dynamic_threshold_v = kandinsky_dynamic_threshold_v
        self.kandinsky_sampler = kandinsky_sampler
        self.kandinsky_ddim_eta = kandinsky_ddim_eta
        self.kandinsky_guidance_scale = kandinsky_guidance_scale
        self.kandinsky_strength = kandinsky_strength
        self.kandinsky_progress = kandinsky_progress
        self.prompt_perturbation = prompt_perturbation
        self.style = style

        self.kandinsky = get_kandinsky2(device, task_type='text2img')
        self.whisper = whisper.load_model(whisper_size)

    def remove_punctuation(self, text):
        return "".join([ch if ch not in string.punctuation else ' ' for ch in text])

    def remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text, flags=re.I)

    def generate_images(self, prompts: list, times: list, title: str = "", artist: str = "", duration: float = 0):
        prompts = [title + " " + artist] + prompts + [title + " " + artist]
        times = [[0, times[0][0]]] + times + [[times[-1][1], duration]]
        pil_imgs = []
        for prompt in prompts:
            style_prompt = prompt + " " + self.style
            pil_img = self.kandinsky.generate_text2img(
                self.remove_multiple_spaces(self.remove_punctuation(style_prompt.lower())),
                batch_size=1, h=self.image_height, w=self.image_width,
                num_steps=self.kandinsky_images_steps,
                denoised_type=self.kandinsky_denoised_type,
                dynamic_threshold_v=self.kandinsky_dynamic_threshold_v,
                sampler=self.kandinsky_sampler, ddim_eta=self.kandinsky_ddim_eta,
                guidance_scale=self.kandinsky_guidance_scale
            )
            pil_imgs.append(pil_img[0])
        scenario = []
        texts = []
        if self.with_img2img_transition:
            frames = []
            for i in range(len(times)):
                frames.append(int((times[i][0] * self.fps + times[i][1] * self.fps) / 2))
            x_average = 0
            for i in range(len(times)):
                x_average += (times[i][1] - times[i][0]) * self.fps
            x_average /= len(times)
            x_average /= 2
            x_average = round(x_average)
            y = 0
            for i in range(len(prompts)):
                while len(texts) < self.fps * times[i][0]:
                    texts.append("")
                peaks = []
                if i > 0:
                    x = frames[i]
                    if 2 * x_average > x - y:
                        peaks = [(x + y) // 2]
                    else:
                        length = x_average
                        w = y + length
                        while w < x:
                            peaks.append(w)
                            w += length
                scenario += [{
                    'frame': frames[i], 'image': pil_imgs[i], 'prompt': prompts[i],
                    'perturbation_peaks': peaks,
                    'prompt_perturbation': self.prompt_perturbation,
                }]
                while len(texts) < self.fps * times[i][1]:
                    texts.append(prompts[i])
                y = frames[i]
        else:
            for i in range(len(prompts)):
                last = 0
                while last <= self.fps * (times[i][1] - times[i][0]):
                    scenario.append((pil_imgs[i], prompts[i]))
                    last += 1
            return scenario
        all_images = self.kandinsky.generate_img2img_flow(
            scenario,
            strength=self.kandinsky_strength, num_steps=self.kandinsky_flow_steps,
            guidance_scale=self.kandinsky_guidance_scale, progress=self.kandinsky_progress,
            dynamic_threshold_v=self.kandinsky_dynamic_threshold_v, denoised_type=self.kandinsky_denoised_type,
            sampler=self.kandinsky_sampler, ddim_eta=self.kandinsky_ddim_eta
        )
        ans = []
        for i in range(len(texts)):
            ans.append((all_images[i], texts[i]))

        return ans

    def generate_alignment(self, song_file: str, lyrics_str: str, language: str, duration: float):
        options = dict(language=language, beam_size=self.whisper_beam_size,
                       best_of=self.whisper_beam_size)
        transcribe_options = dict(task="transcribe", **options)
        transcription = self.whisper.transcribe(song_file, **transcribe_options)
        return align_segments(transcription, lyrics_str, duration)

    def get_song_and_lyrics(self, query: str, song_file: str, ya_music_token: str, genius_token: str):
        lyrics, language = get_lyrics(query, song_file, ya_music_token, genius_token)
        return lyrics, language

    def separate_vocals(self, song_file: str, out_file: str, whisper_sample_rate: int):
        separate_vocals(song_file=song_file, out_file=out_file,
                        whisper_sample_rate=whisper_sample_rate)

    def add_caption_to_image(self, img, caption):
        img = deepcopy(img)
        MAX_W, MAX_H = self.image_width, self.image_height
        l = int(0.08 * MAX_W)
        p = textwrap.wrap(caption, width=int(self.image_width // (l * 0.65)))

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', l)

        current_h, pad = int(0.7 * MAX_H), int(0.02 * MAX_H)
        for line in p:
            w, h = draw.textsize(line, font=font)
            draw.text(((MAX_W - w) // 2, current_h), line, fill='white', font=font,
                      stroke_width=2, stroke_fill='black')
            current_h += h + pad
        return img

    def create_video_clip(self, images_with_texts, song_file, video_file):
        clip = []
        for image, text in images_with_texts:
            clip.append(
                ImageClip(
                    np.array(self.add_caption_to_image(image, text))
                ).set_duration(1 / self.fps)
            )

        videoclip = concatenate_videoclips(clip, method='compose')
        audioclip = AudioFileClip(song_file)
        new_audioclip = CompositeAudioClip([audioclip])
        videoclip.audio = new_audioclip
        videoclip.write_videofile(video_file, fps=self.fps, audio_codec='aac')
