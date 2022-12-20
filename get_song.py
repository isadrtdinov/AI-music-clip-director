# -*- coding: utf-8 -*-

from yandex_music import Client
from yandex_music.track.track import Track
from lyricsgenius import Genius



def get_lyrics_from_genius(query, genius_token):
    genius = Genius(genius_token)

    song = genius.search_song(query)
    if song:
        text = song.lyrics
        lyrics = ""
        last_index = text.find(']')
        while True:
            if last_index == -1:
                break
            index = text.find('[', last_index)
            if index == -1:
                index = len(text)
            lyrics += text[last_index + 1:index]
            last_index = text.find(']', index)
        if lyrics[-5:] == 'Embed':
            lyrics = lyrics[:-5]
        while lyrics[-1].isdigit() or lyrics[-2].isdigit():
            lyrics = lyrics[:-1]
        return lyrics
    else:
        raise Exception("LyricsNotFoundException")


def get_lyrics(query, song_file, ya_music_token, genius_token):
    client = Client(ya_music_token).init()
    search_result = client.search(query)

    track = search_result.best['result']
    if type(track) == Track:
        track.download(song_file)

        if track.lyrics_available:
            supp = track.get_supplement()
            lyrics = supp['lyrics']['full_lyrics']
        else:
            lyrics = get_lyrics_from_genius(query, genius_token)
    else:
        raise Exception("TrackNotFoundException")
    cnteng = 0
    cntru = 0
    for i in lyrics:
        if ord(i.lower()) >= ord("a") and ord(i.lower()) <= ord("z"):
            cnteng += 1
        elif ord(i.lower()) >= ord("а") and ord(i.lower()) <= ord("я"):
            cntru += 1
    if cnteng > cntru:
        return lyrics, "English"
    else:
        return lyrics, "Russian"


