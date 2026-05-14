import sys
import colorama
from util.manager import process_file, process_folder
from util import load_cache, write_cache


def print_help():
    print(f"""
python {colorama.Fore.CYAN + colorama.Style.DIM}main.py {colorama.Fore.BLUE}[command] [arg]{colorama.Style.RESET_ALL}
    
possible commands:
    - file    {colorama.Style.DIM}Process a single file (arg: the file to process){colorama.Style.RESET_ALL}
    - folder  {colorama.Style.DIM}Process a folder recursively (arg: the folder to process){colorama.Style.RESET_ALL}
    - help    {colorama.Style.DIM}Print this help message (no args){colorama.Style.RESET_ALL}
""")


if sys.argv[1] == "file":
    load_cache()
    process_file(sys.argv[2])
    write_cache()
elif sys.argv[1] == "folder":
    load_cache()
    process_folder(sys.argv[2])
    write_cache()
elif sys.argv[1] == "help" or sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print_help()
else:
    print(
        colorama.Fore.RED + colorama.Style.DIM + "Invalid argument",
        sys.argv[1],
        colorama.Style.RESET_ALL,
    )
    print_help()
    exit(1)
