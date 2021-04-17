import csv
import json
from joblib import Parallel, delayed
import subprocess
import time
import os
from scipy.io import wavfile
from pydub import AudioSegment

def get_clip(id, start, stop, folder):
    print('Downloading {}'.format(id))
    command = ["youtube-dl", "-g", "https://www.youtube.com/watch?v={}".format(id)]
    res = subprocess.run(command, capture_output=True)
    url = str(res.stdout).split("\\n")
    if len(url) < 2:
        return -1 # video unavailable
    else:
        url = url[1]

    if 'mime=audio%2Fmp4' in url:
        command = ['ffmpeg', '-i', url, '-ss', time.strftime('%H:%M:%S', time.gmtime(start)), '-to', time.strftime('%H:%M:%S', time.gmtime(stop)), '-c', 'copy', folder + '/{}.m4a'.format(id), '-y']
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

# Determines if an item is labeled label, returns bool
def check_label(item, label):
    label in item["positive_labels"]

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
    if not os.path.isdir(folder):
        os.mkdir(folder)
    params = gen_params(meta, labels, folder)
    def gclip(p):
        return get_clip(*p)
    Parallel(n_jobs=-1, verbose=50)(delayed(gclip)(p) for p in params)

def download_meta_serial(meta, labels, folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    params = gen_params(meta, labels, folder)
    def gclip(p):
        return get_clip(*p)
    for p in params:
        gclip(p)

# Searches through the ontology to find the name associated with the id
def get_label_name_from_id(ontology, ontology_id):
    for item in ontology:
        if item["id"] == ontology_id:
            return item["name"]

# Searches through the ontology to find the id associated with the name
def get_label_id_from_name(ontology, ontology_name):
    for item in ontology:
        if item["name"].upper() == ontology_name.upper():
            return item["id"]

# Returns new meta object with only items that have at least one of the given label_ids
def filter_labels(meta, label_ids):
    rt = list()
    for item in meta:
        for label_id in label_ids:
            if label_id in item["positive_labels"]:
                rt.append(item)
                break
    return rt

# Load the dataset into memory, shows warning for missing files
def load_dataset(folder, meta):
    rt = list()
    for item in meta:
        path = folder + item["ytid"] + ".m4a"
        try:
            data = AudioSegment.from_file(path)
            rt.append((item, data))
        except FileNotFoundError:
            print("WARNING:", path, "was not found")
    return rt

# Load the dataset into memory, shows warning for missing files
def load_dataset_wav(folder, meta):
    rt = list()
    for item in meta:
        path = folder + item["ytid"] + ".wav"
        try:
            (sample_rate, data) = wavfile.read(path)
            rt.append((sample_rate, data))
        except FileNotFoundError:
            print("WARNING:", path, "was not found")
    return rt

def load_dataset_labeled(folder, meta, label_id):
    meta = filter_labels(meta, [label_id])
    return load_dataset(folder, meta)

def load_dataset_multiple_labels(folder, meta, label_ids):
    meta = filter_labels(meta, label_ids)
    return load_dataset(folder, meta)


if __name__ == "__main__":
    eval = load_csv("google_audioset_meta/eval_segments.csv")
    balanced = load_csv("google_audioset_meta/balanced_train_segments.csv")
    unbalanced = load_csv("google_audioset_meta/unbalanced_train_segments.csv")
    ontology = load_ontology()

    classes = ["hammer", "drill"]
    labels = [get_label_id_from_name(ontology, cl) for cl in classes]
    download_meta(eval, labels, 'data/eval')
    download_meta(balanced, labels, 'data/balanced')
    download_meta(unbalanced, labels, 'data/unbalanced')

