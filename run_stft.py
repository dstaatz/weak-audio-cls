#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
# from scipy.linalg import svd

from import_google_audioset import *
from pydub.utils import mediainfo

from sklearn.decomposition import PCA

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
metadata = [audio[i][0] for i in range(len(audio))]
data = [np.array(audio[i][1].set_channels(1).get_array_of_samples()) for i in range(len(audio))]
freqs = [int(mediainfo("data/balanced/" + item["ytid"] + ".m4a")['sample_rate']) for item in metadata]
stfts = [stft(data[i], fs=freqs[i], nperseg=512, nfft=1024) for i in range(len(data))]
# svds = []
# for i in range(len(stfts)):
#     print("Calculating svd {} of shape {}".format(metadata[i]['ytid'], str(stfts[i][2].shape)))
#     svds.append(svd(stfts[i][2]))
#
# print("Sample frequencies:", stfts[0][0].shape)
# print("Segment times:", stfts[0][1].shape)
# print("STFT:", stfts[0][2].shape)
#
# print("U matrix:", svds[0][0].shape)
# print("Sigular values:", svds[0][1].shape)
# print("Vh matrix:", svds[0][2].shape)

# print(stfts[0][0])
# print()
# print(stfts[0][2][:,0])

def plot_stft(i):
    # plot the  stft
    # print(np.abs(stfts[i][2]).min())
    # print(np.abs(stfts[i][2]).max())
    plt.pcolormesh(stfts[i][1], stfts[i][0], np.abs(stfts[i][2]), vmin=0, vmax=250, shading="auto")
    plt.title("STFT Magnitude of %s" % (metadata[i]["ytid"]))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()


pca = PCA(n_components=100)
xvals = [np.abs(stft[2]).T for stft in stfts]
# X = np.concatenate([np.abs(stft[2]) for stft in stfts], axis=1)
X = np.concatenate(xvals, axis=0)
pca.fit(X)
t_xvals = [pca.transform(x) for x in xvals]
t_X = pca.transform(X)

