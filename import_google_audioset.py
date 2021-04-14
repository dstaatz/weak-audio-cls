#!/usr/bin/python

from __future__ import unicode_literals
import youtube_dl

import csv
import json

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

# Downloads all the youtube video ids as mp3s saved to the given folder
def bulk_download(folder, ids):

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": folder + "%(id)s.%(ext)s",
        "ignoreerrors": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192",
        }]
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(ids)

# Downloads all ids in meta_data that have the given label_id to the given folder
def download_labeled(meta_data, label_id, folder):
    ids = list()
    for d in meta_data:
        if label_id in d["positive_labels"]:
            ids.append(d["ytid"])
    bulk_download(folder, ids)

# Downloads all ids in meta_data that have a label in the list label_ids to the given folder
def download_multiple_labels(meta_data, label_ids, folder):
    ids = list()
    for d in meta_data:
        for label_id in label_ids:
            if label_id in d["positive_labels"]:
                ids.append(d["ytid"])
                break
    bulk_download(folder, ids)

def gen_params(data, labels, folder):
    params = []
    for d in data:
        if len(list(set(labels) & set(d["positive_labels"]))) > 0:
            params.append((d["ytid"], d['start_seconds'], d['end_seconds'], folder))
    return params

# Test function
if __name__ == "__main__":
    test = load_csv("google_audioset_meta/eval_segments.csv")
    # train = load_csv("google_audioset_meta/balanced_train_segments.csv")
    # train_un = load_csv("google_audioset_meta/unbalanced_train_segments.csv")

    labels = ["/m/03l9g"]
    # download_multiple_labels(test, labels, "data/test/")
    from clip_download import get_clip
    from joblib import Parallel, delayed
    params = gen_params(test, labels, 'data/test')
    def gclip(p):
        return get_clip(*p)
    Parallel(n_jobs=-1, verbose=50)(delayed(gclip)(p) for p in params)

    # data1 = load_csv("google_audioset_meta/eval_segments.csv")
    # for i in range(0, 10):
    #     print(data1[i])
    # print()
    #
    # data2 = load_csv("google_audioset_meta/balanced_train_segments.csv")
    # for i in range(0, 5):
    #     print(data2[i])
    # print()
    #
    # data3 = load_csv("google_audioset_meta/unbalanced_train_segments.csv")
    # for i in range(0, 5):
    #     print(data3[i])
    # print()
    #
    # ontology = load_ontology()
    # print(ontology[0])
    # print()
    #
    # ids = [x["ytid"] for x in data1]
    # folder = "data/test/"
    # print(ids[30:50])
    # bulk_download(folder, ids[30:50])
    #
    # labels = ["/m/04rlf", "/t/dd00004", "/t/dd00005", "/m/03l9g"]
    # # download_labeled(data1[0:10], labels[0], folder)
    # download_multiple_labels(data1[0:10], labels, folder)

