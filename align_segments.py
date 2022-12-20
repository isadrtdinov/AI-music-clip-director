import codecs
import pickle
from fuzzywuzzy import fuzz


def align_segments(transcription: , lyrics: str):
    objects = []
    with (open(transcription_pickle_file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    segments = []
    times = []
    for segment in objects[0]['segments']:
        segments.append(segment['text'].lower().strip())
        segments[-1] = segments[-1].replace('(', '')
        segments[-1] = segments[-1].replace(')', '')
        times.append([segment['start'], segment['end']])

    strings = ""
    starts = []
    ends = []
    lens = [0]
    with codecs.open(lyrics_file, "r") as f:
        s = str(f.read()).strip()
        for i in s.strip().split('\n'):
            t = i.lower()
            t = t.replace('(', '')
            t = t.replace(')', '')
            if t:
                for word in t.split():
                    starts.append(len(strings))
                    ends.append(starts[-1] + len(word) - 1)
                    if len(lens) == 0:
                        lens.append(ends[-1] - starts[-1] + 1)
                    else:
                        lens.append(ends[-1] - starts[-1] + 1 + lens[-1] + 1)
                    strings += word + ' '

    best = []
    for i in range(len(segments)):
        best_distance = 0
        bestl = 0
        bestr = 0
        for ql in range(0, len(starts) + 1):
            qr = ql
            while qr <= len(ends) and lens[qr] - lens[ql - 1] <= len(segments[i]) + 10:
                l = starts[ql - 1]
                r = ends[qr - 1]
                distance = fuzz.ratio(segments[i], strings[l:r + 1])
                if distance > best_distance or (distance == best_distance and r - l + 1 < bestr - bestl + 1):
                    best_distance = distance
                    bestr = r
                    bestl = l
                qr += 1
        best.append((bestl, bestr))
    ans = []
    for i in range(len(segments)):
        ans.append({'text': strings[best[i][0]:best[i][1] + 1], 'start': times[i][0], 'end': times[i][1]})
    return ans
