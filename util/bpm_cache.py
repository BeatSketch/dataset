import time
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


def get_bpm_for_file_list(files: list[str]) -> list[str]:
    """Get BPM for the file from its filename.
    For that to work, the file needs to have the exact format that the BeatLeader
    mod uses to save the file

    Args:
        files: The files to process

    Returns:
        list of files that were processable
    """
    file_list: list[str] = []
    for file in files:
        hash = file.split("-")[-2]
        if len(hash) == 40:
            bpm = get_bpm_for_song(hash)
            if bpm[1]:
                write_cache()
                time.sleep(0.5)
            if bpm[0] > 0:
                file_list.append(file)
    return file_list


def get_bpm_for_song(song_hash: str) -> tuple[int, bool]:
    downloaded = False
    try:
        bpm_cache[song_hash]
    except KeyError:
        print(
            colorama.Fore.BLUE + colorama.Style.DIM + "Downloading info for song",
            song_hash,
            colorama.Style.RESET_ALL,
        )
        res = requests.get("https://beatsaver.com/api/maps/hash/" + song_hash)
        decoded_json = res.json()
        try:
            bpm_cache[song_hash] = int(decoded_json["metadata"]["bpm"])
            downloaded = True
        except KeyError:
            bpm_cache[song_hash] = -1
    return bpm_cache[song_hash], downloaded


def write_cache():
    with open("bpmcache.json", "w") as f:
        f.write(json.dumps(bpm_cache))
