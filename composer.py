import whisper
from kandinsky2 import get_kandinsky2
from align_segments import align_segments
from get_song import get_lyrics
from separate_vocals import separate_vocals
from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import os
import numpy as np


class ClipDirector(object):
    def __init__(self, device,
                 whisper_size: str = 'medium', whisper_beam_size: int = 10,
                 fps: int = 20, image_height: int = 512, image_width: int = 512, with_img2img_transition: bool = True,
                 kandinsky_images_steps: int = 75, kandinsky_flow_steps: int = 100,
                 kandinsky_denoised_type: str = "dynamic_threshold", kandinsky_dynamic_threshold_v: float = 99.5,
                 kandinsky_sampler: str = "kandinsky_sampler", kandinsky_ddim_eta: float = 0.05,
                 kandinsky_guidance_scale: float = 7, kandinsky_strength: float = 0.9, kandinsky_progress: bool = True,
                 prompt_perturbation: float = 0.25):
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

        self.kandinsky = get_kandinsky2(device, task_type='text2img')
        self.whisper = whisper.load_model(whisper_size)

    def generate_images(self, prompts: list, times: list):
        pil_imgs = []
        for prompt in prompts:
            pil_img = self.kandinsky.generate_text2img(prompt, batch_size=1, h=self.image_height, w=self.image_width,
                                                       num_steps=self.kandinsky_images_steps,
                                                       denoised_type=self.kandinsky_denoised_type,
                                                       dynamic_threshold_v=self.kandinsky_dynamic_threshold_v,
                                                       sampler=self.kandinsky_sampler, ddim_eta=self.kandinsky_ddim_eta,
                                                       guidance_scale=self.kandinsky_guidance_scale)
            pil_imgs.append(pil_img[0])
        scenario = []
        texts = []
        if self.with_img2img_transition:
            last = 0
            s = 0
            for i in range(len(prompts)):
                s += len(prompts[i])
            s /= len(prompts)
            s /= 2

            for i in range(len(prompts)):
                peaks = []

                scenario += [{
                    'frame': int(last + self.fps * times[i] / 2), 'image': pil_imgs[i], 'prompt': prompts[i],
                    'perturbation_peaks': [int(last + self.fps * times[i] / 2)],
                    'prompt_perturbation': self.prompt_perturbation,
                }]
                for j in range(self.fps * times[i]):
                    texts.append(prompts[i])
                last += int(self.fps * times[i])
        else:
            for i in range(len(prompts)):
                last = 0
                while last <= self.fps * times[i]:
                    scenario.append((pil_imgs[i], prompts[i]))
                    last += 1
            return scenario
        all_images = self.kandinsky.generate_img2img_flow(scenario,
                                                          strength=self.kandinsky_strength,
                                                          num_steps=self.kandinsky_flow_steps,
                                                          guidance_scale=self.kandinsky_guidance_scale,
                                                          progress=self.kandinsky_progress,
                                                          dynamic_threshold_v=self.kandinsky_dynamic_threshold_v,
                                                          denoised_type=self.kandinsky_denoised_type,
                                                          sampler=self.kandinsky_sampler,
                                                          ddim_eta=self.kandinsky_ddim_eta)
        ans = []
        for i in range(len(texts)):
            ans.append((all_images[i], texts[i]))

        return ans

    def generate_alignment(self, song_file: str, lyrics_str: str, language: str):
        options = dict(language=language, beam_size=self.whisper_beam_size,
                       best_of=self.whisper_beam_size)
        transcribe_options = dict(task="transcribe", **options)
        transcription = self.whisper.transcribe(song_file, **transcribe_options)
        return align_segments(transcription, lyrics_str)

    def get_song_and_lyrics(self, query: str, song_file: str, ya_music_token: str, genius_token: str):
        lyrics, language = get_lyrics(query, song_file, ya_music_token, genius_token)
        return lyrics, language

    def separate_vocals(self, song_file: str, out_file: str, out_sample_rate: int):
        separate_vocals(song_file=song_file, out_file=out_file, out_sample_rate=out_sample_rate)

    def add_caption_2_image(self, img, caption):
        MAX_W, MAX_H = self.image_width, self.image_height
        l = int(0.04 * MAX_W)
        p = textwrap.wrap(caption, width=int(self.image_width // (l * 0.65)))

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', l)

        current_h, pad = int(0.7 * MAX_H), int(0.02 * MAX_H)
        for line in p:
            w, h = draw.textsize(line, font=font)
            draw.text(((MAX_W - w) // 2, current_h), line, fill='white', font=font, stroke_width=2, stroke_fill='black')
            current_h += h + pad
        return img
    def create_video_clip(self, images_with_texts):
        all_images = []
        for i in range(len(images_with_texts)):
            all_images.append(self.add_caption_2_image(images_with_texts[i][0], images_with_texts[i][1]))
        cv2_images = [np.array(image)[..., ::-1].copy() for image in all_images]
        height, width, layers = cv2_images[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter('video.mp4', fourcc, self.fps, (height, width))

        for image in cv2_images:
            video.write(image)

        cv2.destroyAllWindows()
        video.release()
