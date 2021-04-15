#!/usr/bin/python

# Python imports
import csv
import json
import os
import subprocess

# Package imports
import youtube_dl
import ffmpeg
from scipy.io import wavfile

# Convert 15 seconds to "00:00:15.00"
def secs_to_time_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d.00" % (h, m, s)

# Returns list of dict of labels
def load_meta_csv(filename):
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

# Downloads all the youtube video from the meta-data and converts to wav files to the given folder
def bulk_download(folder, meta, clip=True):

    ids = [x["ytid"] for x in meta]

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": folder + "%(id)s.%(ext)s", # TODO: Could make more robust
        "ignoreerrors": True,
        "quiet": False,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }]
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(ids)

    if clip:
        clip_all(folder, meta)

# Returns new meta object with only items that have at least one of the given label_ids
def filter_labels(meta, label_ids):
    rt = list()
    for item in meta:
        for label_id in label_ids:
            if label_id in item["positive_labels"]:
                rt.append(item)
                break
    return rt

# Downloads all ids in meta that have the given label_id to the given folder
def download_labeled(folder, meta, label_id, clip=True):
    meta = filter_labels(meta, [label_id])
    bulk_download(folder, meta, clip)

# Downloads all ids in meta that have a label in the list label_ids to the given folder
def download_multiple_labels(folder, meta, label_ids, clip=True):
    meta = filter_labels(meta, label_ids)
    bulk_download(folder, meta, clip)

# Use ffmpeg to clip a downloaded file in the given folder
# item_meta is the dictionary in meta that gives the ytid, and start and end times
# Throws FileNotFoundError
def clip_item(folder, item_meta):
    path = folder + item_meta["ytid"] + ".wav"

    if not os.path.isfile(path):
        raise FileNotFoundError

    # File exists, use ffmpeg to convert in place
    start_time = secs_to_time_str(item_meta["start_seconds"])
    capture_time = secs_to_time_str(item_meta["end_seconds"] - item_meta["start_seconds"])

    # Copy the file to a temporary location
    temp_path = folder + "temp.wav"
    subprocess.run(["cp", path, temp_path])

    (
        ffmpeg
        .input(temp_path, ss=start_time, t=capture_time)
        .output(path)
        .overwrite_output()
        .run()
    )

    # Remove the temporary file
    subprocess.run(["rm", temp_path])

# clip all the files in the given folder using the meta data
def clip_all(folder, meta):
    for item in meta:
        try:
            clip_item(folder, item)
        except FileNotFoundError:
            pass

# Load the dataset into memory, shows warning for missing files
def load_dataset(folder, meta):
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

# Test function
if __name__ == "__main__":

    meta1 = load_meta_csv("google_audioset_meta/eval_segments.csv")

    # meta2 = load_meta_csv("google_audioset_meta/balanced_train_segments.csv")

    # # meta3 = load_meta_csv("google_audioset_meta/unbalanced_train_segments.csv")

    # ontology = load_ontology()
    # print(get_label_name_from_id(ontology, "/m/03l9g"))
    # print(get_label_id_from_name(ontology, "Hammer"))

    folder = "data/test/"
    # download_labeled(folder, meta1, "/m/03l9g")
    data = load_dataset_labeled(folder, meta1, "/m/03l9g")
    

    # ids = [x["ytid"] for x in meta1]
    # folder = "data/eval_segments/"
    # bulk_download(folder, ids)

    # ids = [x["ytid"] for x in meta2]
    # folder = "data/balanced_train_segments/"
    # bulk_download(folder, ids)

    # # ids = [x["ytid"] for x in meta3]
    # # folder = "data/unbalanced_train_segments/"
    # # bulk_download(folder, ids)

