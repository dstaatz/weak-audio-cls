#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
from scipy.linalg import svd

from import_google_audioset import *


## Parameters

fs = 44100 # Sampling frequency (hz)


# eval_meta = load_csv("google_audioset_meta/eval_segments.csv")
balanced_meta = load_csv("google_audioset_meta/balanced_train_segments.csv")
# unbalanced_meta = load_csv("google_audioset_meta/unbalanced_train_segments.csv")
ontology = load_ontology()

# Download a dataset
label = get_label_id_from_name(ontology, "hammer")
# download_labeled(folder, balanced_meta, label)
meta = filter_labels(balanced_meta, [label])

# Load in a dataset and perform feature extraction
audio = load_dataset("data/balanced/", meta)
labels = [audio[i][0] for i in range(len(audio))]
data = [np.array(audio[i][1].get_array_of_samples()) for i in range(len(audio))]
stfts = [stft(data[i], fs=fs, nperseg=512, nfft=1024) for i in range(len(data))]
svds = []
for i in range(len(stfts)):
    print("Calculating svd number %i of shape %s" % (i, str(stfts[i][2].shape)))
    svds.append(svd(stfts[i][2]))

print("Sample frequencies:", stfts[0][0].shape)
print("Segment times:", stfts[0][1].shape)
print("STFT:", stfts[0][2].shape)

print("U matrix:", svds[0][0].shape)
print("Sigular values:", svds[0][1].shape)
print("Vh matrix:", svds[0][2].shape)

# print(stfts[0][0])
# print()
# print(stfts[0][2][:,0])

# plot the first stft
print(np.abs(stfts[0][2]).min())
print(np.abs(stfts[0][2]).max())
plt.pcolormesh(stfts[0][1], stfts[0][0], np.abs(stfts[0][2]), vmin=0, vmax=250, shading="auto")
plt.title("STFT Magnitude of %s" % (labels[0]["ytid"]))
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.show()

# plot the second stft
print(np.abs(stfts[1][2]).min())
print(np.abs(stfts[1][2]).max())
plt.pcolormesh(stfts[1][1], stfts[1][0], np.abs(stfts[1][2]), vmin=0, vmax=250, shading="auto")
plt.title("STFT Magnitude of %s" % (labels[1]["ytid"]))
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.show()

