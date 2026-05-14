import colorama
import requests
import json

global bpm_cache
bpm_cache: dict[str, int] = {}


def load_cache():
    global bpm_cache
    try:
        with open("bpmcache.json", "r") as f:
            bpm_cache = json.loads(f.read())
            print(
                colorama.Fore.GREEN + colorama.Style.DIM + "Loaded cache file for BPM",
                colorama.Style.RESET_ALL,
            )
    except FileNotFoundError:
        print(
            colorama.Fore.RED + colorama.Style.DIM + "No cache file for BPM found",
            colorama.Style.RESET_ALL,
        )


def get_bpm_for_song(song_hash: str):
    try:
        bpm_cache[song_hash]
    except KeyError:
        print(
            colorama.Fore.BLUE + colorama.Style.DIM + "Downloading info for song",
            song_hash,
            colorama.Style.RESET_ALL,
        )
        res = requests.get("https://beatsaver.com/api/maps/hash/" + song_hash)
        bpm_cache[song_hash] = int(res.json()["metadata"]["bpm"])
    return bpm_cache[song_hash]


def write_cache():
    with open("bpmcache.json", "w") as f:
        f.write(json.dumps(bpm_cache))
