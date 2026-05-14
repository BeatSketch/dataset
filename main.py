import sys
from util.manager import process_file
from util import write_cache

# TODO: Use folder instead of specific file and process all files in the specified folder

# TODO: Change this to do batch processing instead
process_file(sys.argv[1])


# Make sure to leave this in to persist the cache and further reduce load on BeatSaver
write_cache()
