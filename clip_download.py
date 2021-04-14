import subprocess
import time

def get_clip(id, start, stop, folder):
    # print('Downloading {}'.format(id))
    command = ["youtube-dl", "-g", "https://www.youtube.com/watch?v={}".format(id)]
    res = subprocess.run(command, capture_output=True)
    url = str(res.stdout).split("\\n")
    if len(url) < 2:
        return -1 # video unavailable
    else:
        url = url[1]
    command = ['ffmpeg', '-i', url, '-ss', time.strftime('%H:%M:%S', time.gmtime(start)), '-to', time.strftime('%H:%M:%S', time.gmtime(stop)), '-c', 'copy', folder + '/{}.m4a'.format(id)]
    res = subprocess.run(command, capture_output=True)
    return 1