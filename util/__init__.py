import requests
import json

bpm_cache: dict[str, int] = {}

try:
    with open("bpmcache.json", "r") as f:
        bpm_cache = json.loads(f.read())
        print("Loaded cache file for BPM")
except FileNotFoundError:
    print("No cache file for BPM found")


def get_bpm_for_song(song_hash: str):
    try:
        bpm_cache[song_hash]
    except KeyError:
        print("Downloading info for song", song_hash)
        res = requests.get("https://beatsaver.com/api/maps/hash/" + song_hash)
        bpm_cache[song_hash] = int(res.json()["metadata"]["bpm"])
    return bpm_cache[song_hash]


def write_cache():
    with open("bpmcache.json", "w") as f:
        f.write(json.dumps(bpm_cache))
