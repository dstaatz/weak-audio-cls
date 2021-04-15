import csv
import json
from joblib import Parallel, delayed
import subprocess
import time
import os

def get_clip(id, start, stop, folder):
    # print('Downloading {}'.format(id))
    command = ["youtube-dl", "-g", "https://www.youtube.com/watch?v={}".format(id)]
    res = subprocess.run(command, capture_output=True)
    url = str(res.stdout).split("\\n")
    if len(url) < 2:
        return -1 # video unavailable
    else:
        url = url[1]

    if 'mime=audio%2Fmp4' in url:
        command = ['ffmpeg', '-i', url, '-ss', time.strftime('%H:%M:%S', time.gmtime(start)), '-to', time.strftime('%H:%M:%S', time.gmtime(stop)), '-c', 'copy', folder + '/{}.m4a'.format(id)]
        res = subprocess.run(command, capture_output=True)
        return 1

    else:
        command = ['ffmpeg', '-i', url, '-c', 'copy', folder + '/temp_{}.m4a'.format(id)]
        res = subprocess.run(command, capture_output=True)
        command = ['ffmpeg', '-i', folder + '/temp_{}.m4a'.format(id), '-c', 'copy', folder + '/{}.m4a'.format(id)]
        res = subprocess.run(command, capture_output=True)
        os.remove(folder + '/temp_{}.m4a'.format(id))
        return 1

# Returns list of dict of labels
def load_csv(filename):
    data = list()
    with open(filename) as file:
        r = csv.reader(file)

        # Skip first 3 lines
        r.__next__(), r.__next__(), r.__next__()

        # iterate through rest
        for line in r:
            d = dict()
            d["ytid"] = line[0]
            d["start_seconds"] = float(line[1])
            d["end_seconds"] = float(line[2])
            positive_labels = ','.join(line[3:])
            d["positive_labels"] = positive_labels[2:len(positive_labels)-1].split(',')
            data.append(d)
    return data

# Returns list of dict of label information
def load_ontology():
    with open("google_audioset_meta/ontology.json") as file:
        return json.load(file)

def gen_params(data, labels, folder):
    params = []
    for d in data:
        if len(list(set(labels) & set(d["positive_labels"]))) > 0:
            params.append((d["ytid"], d['start_seconds'], d['end_seconds'], folder))
    return params

def download_meta(meta, labels, folder):
    params = gen_params(meta, labels, folder)
    def gclip(p):
        return get_clip(*p)
    Parallel(n_jobs=-1, verbose=50)(delayed(gclip)(p) for p in params)

if __name__ == "__main__":
    eval = load_csv("google_audioset_meta/eval_segments.csv")
    balanced = load_csv("google_audioset_meta/balanced_train_segments.csv")
    unbalanced = load_csv("google_audioset_meta/unbalanced_train_segments.csv")

    labels = ["/m/03l9g"]
    download_meta(eval, labels, 'data/eval')
    download_meta(balanced, labels, 'data/balanced')
    download_meta(unbalanced, labels, 'data/unbalanced')

