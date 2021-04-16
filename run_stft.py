#!/usr/bin/python

from matplotlib import pyplot as plt

from import_google_audioset import *


meta1 = load_csv("google_audioset_meta/eval_segments.csv")

folder = "data/eval/"
# download_labeled(folder, meta1, "/m/03l9g")
data = load_dataset_labeled(folder, meta1, "/m/03l9g")
print(data[0])




