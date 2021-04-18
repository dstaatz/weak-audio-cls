#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
# from scipy.linalg import svd

from import_google_audioset import *
from pydub.utils import mediainfo
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm

# eval_meta = load_csv("google_audioset_meta/eval_segments.csv")
balanced_meta = load_csv("google_audioset_meta/balanced_train_segments.csv")
# unbalanced_meta = load_csv("google_audioset_meta/unbalanced_train_segments.csv")
ontology = load_ontology()

# Load in a dataset and perform feature extraction
classes = ["hammer", "drill"]
labels = [get_label_id_from_name(ontology, cl) for cl in classes]
meta = filter_labels(balanced_meta, labels)
audio = load_dataset("data/balanced/", meta)
metadata = [audio[i][0] for i in range(len(audio))]
data = [np.array(audio[i][1].set_channels(1).get_array_of_samples()) for i in range(len(audio))]
freqs = [int(mediainfo("data/balanced/" + item["ytid"] + ".m4a")['sample_rate']) for item in metadata]
stfts = [stft(data[i], fs=freqs[i], nperseg=512, nfft=1024) for i in range(len(data))]

y_labels = []
for item in metadata:
    for label in labels:
        if label in item["positive_labels"]:
            y_labels.append(label)
            break # If multiple labels are assigned to the same item, just take the first label

def plot_stft(i):
    # plot the stft
    # print(np.abs(stfts[i][2]).min())
    # print(np.abs(stfts[i][2]).max())
    plt.pcolormesh(stfts[i][1], stfts[i][0], np.abs(stfts[i][2]), vmin=0, vmax=250, shading="auto")
    plt.title("STFT Magnitude of %s" % (metadata[i]["ytid"]))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()


# Do PCA
print("Running PCA")
pca = PCA(n_components=100)
xvals = [np.abs(stft[2]).T for stft in stfts]
# X = np.concatenate([np.abs(stft[2]) for stft in stfts], axis=1)
X = np.concatenate(xvals, axis=0)
pca.fit(X)
t_xvals = [pca.transform(x) for x in xvals]
t_X = pca.transform(X)

# Do K-Means
print("Running K-means")
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(t_X)
onehot = np.eye(n_clusters)
kt_xvals = [onehot[kmeans.predict(x)] for x in t_xvals]
# kt_X = onehot[kmeans.predict(t_X)]

timediffs = [(stft[1][1] - stft[1][0]) for stft in stfts]
timestep = 0.1
binsize = [int(timestep/td) for td in timediffs]

def chunks(lst, n):
    # https://stackoverflow.com/a/312464
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
hist_xvals = [np.array([np.mean(h, axis=0) for h in chunks(x, n)]) for x, n in zip(kt_xvals, binsize)]
num_segments_per_clip = [hist_xval.shape[0] for hist_xval in hist_xvals]

# Train svm
test_Cs = [2**i for i in range(-7, 7)] # paper uses range(-7, 7)
test_gammas = [2**i for i in range(-7, 7)] # paper uses range(-7, 7)

# Hold a certian percent of each class to become the holdout set
train_percent = 0.7
counts = [y_labels.count(label) for label in labels]
train_counts = [round(train_percent * c) for c in counts]

idxs = [[i for i in range(len(y_labels)) if y_labels[i] == label] for label in labels]
train_idxs = sum([idxs[i][:train_counts[i]] for i in range(len(labels))], [])
holdout_idxs = sum([idxs[i][train_counts[i]:] for i in range(len(labels))], [])
train_idxs.sort() # shouldn't really matter
holdout_idxs.sort() # shouldn't really matter

y_train = np.concatenate([[y_labels[idx] for i in range(num_segments_per_clip[idx])] for idx in train_idxs])
y_holdout = np.concatenate([[y_labels[idx] for i in range(num_segments_per_clip[idx])] for idx in holdout_idxs])

X_train = np.concatenate([hist_xvals[idx] for idx in train_idxs])
X_holdout = np.concatenate([hist_xvals[idx] for idx in holdout_idxs])

# Fit to all test_Cs and test_gammas for a radial basis kernel SVM

def train_svm(params):
    
params = gen_params(meta, labels, folder)
def gclip(p):
    return get_clip(*p)
Parallel(n_jobs=-1, verbose=50)(delayed(gclip)(p) for p in params)

clfs = []
for C in test_Cs:
    temp = []
    for gamma in test_gammas:
        print("Training SVM (C = %s, gamma = %s)" % (str(C), str(gamma)))
        clf = svm.SVC(C=C, kernel="rbf", gamma=gamma)
        clf.fit(X_train, y_train)
        temp.append(clf)
    clfs.append(temp)
print()

# Determine the best C and gamma for a radial basis kernel
prob_errs = []
for i in range(len(test_Cs)):
    temp = []
    for j in range(len(test_gammas)):
        print("Error (C = %s, gamma = %s): " % (str(test_Cs[i]), str(test_gammas[j])), end="")
        prob_err = 1 - clfs[i][j].score(X_holdout, y_holdout)
        print(prob_err)
        temp.append(prob_err)
    prob_errs.append(temp)

min_err = min([min(temp) for temp in prob_errs])

min_idx = None
for (i, temp) in enumerate(prob_errs):
    if min_err in temp:
        min_idx = (i, temp.index(min_err))
        break

print()
print("Minimum error:", min_err)
print("Minimum index:", min_idx)

best_C = test_Cs[min_idx[0]]
best_gamma = test_gammas[min_idx[1]]

print("Best C:", best_C)
print("Best gamma:", best_gamma)
