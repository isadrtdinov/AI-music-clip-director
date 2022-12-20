from get_song import get_lyrics
from align_segments import align_segments


query = "Кукла колдуна"
song_file = "song.mp3"
lyrics = get_lyrics(query, song_file)
lyrics_file = "ans.txt"

transcription_pickle_file = "transcription.pickle"
with open(lyrics_file, 'w') as file:
    file.write(lyrics)


ans = align_segments(lyrics_file, transcription_pickle_file)
for i in range(len(ans)):
    print(ans[i]["text"], ans[i]["start"], ans[i]["end"])

