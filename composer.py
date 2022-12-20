import whisper
from kandinsky2 import get_kandinsky2
from align_segments import align_segments


class ClipDirector(object):
    def __init__(self, device,
                 whisper_size: str = 'medium', whisper_beam_size: int = 10,
                 fps: int = 20):
        self.whisper_size = whisper_size
        self.whisper_beam_size = whisper_beam_size

        self.kandinsky = get_kandinsky2(device, task_type='text2img')
        self.whisper = whisper.load_model(whisper_size)

    def generate_images(self):
        pass

    def generate_alignment(self, song_file: str, lyrics_str: str, language: str):
        options = dict(language=language, beam_size=self.whisper_beam_size,
                       best_of=self.whisper_beam_size)
        transcribe_options = dict(task="transcribe", **options)
        transcription = self.whisper.transcribe(song_file, **transcribe_options)
        align_segments()


    def create_video_clip(self):
        pass
