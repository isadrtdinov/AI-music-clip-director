import librosa
import numpy as np
import soundfile as sf


def separate_vocals(song_file: str, out_file: str, whisper_sample_rate: int):
    """
    :param song_file:
    :param out_file:
    :param out_sample_rate:
    :return:
    """
    # load waveform from song file
    waveform, sample_rate = librosa.load(song_file)

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.
    S_full, phase = librosa.magphase(librosa.stft(waveform))
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median,
        metric='cosine', width=int(librosa.time_to_frames(2, sr=sample_rate))
    )

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive. Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    new_waveform = librosa.istft(S_foreground * phase)
    new_waveform = librosa.resample(
        new_waveform, orig_sr=sample_rate, target_sr=whisper_sample_rate
    )
    sf.write(out_file, new_waveform, whisper_sample_rate)
