import codecs
import pickle
from fuzzywuzzy import fuzz


def align_segments(transcription: dict, lyrics: str):

    segments = []
    times = []
    for segment in transcription['segments']:
        segments.append(segment['text'].lower().strip())
        segments[-1] = segments[-1].replace('(', '')
        segments[-1] = segments[-1].replace(')', '')
        times.append([segment['start'], segment['end']])

    strings = ""
    starts = []
    ends = []
    lens = [0]
    song_strings = []
    s = lyrics
    for i in s.strip().split('\n'):
        if i:
            song_strings.append(i)
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

    d = []
    best = []
    from fuzzywuzzy import fuzz
    for i in range(len(segments)):
        # print(i, len(segments))
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
    # for i in range(len(best)):
    #     l, r = best[i]
    #     print(segments[i], '||||', strings[l:r+1])
    mid = 0
    itog = []
    for i in range(len(segments)):
        itog.append(strings[best[i][0]:best[i][1] + 1])
    d = []
    for i in range(len(itog)):
        for j in itog[i].split():
            d.append([j, i])
    last = 0
    ans = []
    itog_times = []
    for i in range(len(song_strings)):
        l = last
        r = last
        s = d[l][0]
        r += 1
        while r < len(d) and (
                fuzz.ratio(song_strings[i], s + " " + d[r][0]) > fuzz.ratio(song_strings[i], s) or fuzz.ratio(
                song_strings[i], s) < 35):
            s += " " + d[r][0]
            r += 1
        r -= 1
        l += 1
        while l <= r and (
                fuzz.ratio(s[len(d[l][0]) + 1:], song_strings[i]) > fuzz.ratio(song_strings[i], s) or fuzz.ratio(
                song_strings[i], s) < 35):
            s = s[len(d[l][0]) + 1:]
            l += 1
        l -= 1
        ans.append("")
        if fuzz.ratio(s, song_strings[i]) < 35:
            itog_times.append([-1, -1])
            continue
        itog_times.append([times[d[l][1]][0], times[d[r][1]][1]])
        for j in range(l, r + 1):
            ans[-1] += d[j][0] + " "
        last = r + 1
    superitog = []
    last = 0
    for i in range(len(song_strings)):
        if itog_times[i] == [-1, -1]:
            superitog.append([last, last])
            continue
        j = i
        while j < len(song_strings) and itog_times[j][0] < itog_times[i][1]:
            j += 1

        superitog.append([max(last, itog_times[i][0]),
                          min(max(last, itog_times[i][0]) + (itog_times[i][1] - itog_times[i][0]) / (j - i),
                              itog_times[i][1])])
        last = min(max(last, itog_times[i][0]) + (itog_times[i][1] - itog_times[i][0]) / (j - i), itog_times[i][1])

    q = []
    for i in range(len(superitog)):
        q.append([song_strings[i], [superitog[i][0], superitog[i][1]]])
    return q

