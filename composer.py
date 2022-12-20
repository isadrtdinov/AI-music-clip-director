import whisper
from kandinsky2 import get_kandinsky2
from align_segments import align_segments
from get_song import get_lyrics
from separate_vocals import separate_vocals


class ClipDirector(object):
    def __init__(self, device,
                 whisper_size: str = 'medium', whisper_beam_size: int = 10,
                 fps: int = 20):
        self.whisper_size = whisper_size
        self.whisper_beam_size = whisper_beam_size

        self.kandinsky = get_kandinsky2(device, task_type='text2img')
        self.whisper = whisper.load_model(whisper_size)

    def generate_images(self, prompts: list):
        pil_imgs = []
        for prompt in prompts:
            pil_img = self.kandinsky.generate_text2img(prompt, batch_size=1, h=512, w=512, num_steps=75,
                                                       denoised_type='dynamic_threshold', dynamic_threshold_v=99.5,
                                                       sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=7)
            pil_imgs.append(pil_img[0])
        scenario = []
        for i in range(len(prompts)):
            scenario += [{
                'frame': i * 30, 'image': pil_imgs[i], 'prompt': prompts[i],
                'perturbation_peaks': [i * 30 + 15], 'prompt_perturbation': 0.25,
            }]
        all_images = self.kandinsky.generate_img2img_flow(scenario,
                                                          strength=0.9, num_steps=100, guidance_scale=7, progress=True,
                                                          dynamic_threshold_v=99.5, denoised_type='dynamic_threshold',
                                                          sampler='ddim_sampler', ddim_eta=0.05)
        return all_images

    def generate_alignment(self, song_file: str, lyrics_str: str, language: str):
        options = dict(language=language, beam_size=self.whisper_beam_size,
                       best_of=self.whisper_beam_size)
        transcribe_options = dict(task="transcribe", **options)
        transcription = self.whisper.transcribe(song_file, **transcribe_options)
        return align_segments(transcription, lyrics_str)

    def get_song_and_lyrics(self, query: str, song_file: str, ya_music_token: str, genius_token: str):
        lyrics, language = get_lyrics(query, song_file, ya_music_token, genius_token)
        return lyrics, language

    def separae_vocals(self, song_file: str, out_file: str, out_sample_rate: int):
        separate_vocals(song_file=song_file, out_file=out_file, out_sample_rate=out_sample_rate)

    def create_video_clip(self):
        pass
