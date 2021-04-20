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
import noisereduce as nr

import cv2
import subprocess

# eval_meta = load_csv("google_audioset_meta/eval_segments.csv")
balanced_meta = load_csv("google_audioset_meta/balanced_train_segments.csv")
# unbalanced_meta = load_csv("google_audioset_meta/unbalanced_train_segments.csv")
ontology = load_ontology()

# Load in a dataset and perform feature extraction
classes = ["hammer", "drill"]
# classes = ['hammer']
labels = [get_label_id_from_name(ontology, cl) for cl in classes]
meta = filter_labels(balanced_meta, labels)
audio = load_dataset("data/balanced/", meta)
metadata = [audio[i][0] for i in range(len(audio))]
data = [np.array(audio[i][1].set_channels(1).get_array_of_samples()) for i in range(len(audio))]
data = [np.array([float(f) for f in d]) for d in data]
freqs = [int(mediainfo("data/balanced/" + item["ytid"] + ".m4a")['sample_rate']) for item in metadata]

print("Filtering Noise")
green_noise = np.array(AudioSegment.from_file("green_noise.m4a").set_channels(1).get_array_of_samples())
green_noise = np.array([float(f) for f in green_noise])
data = [nr.reduce_noise(data[i], green_noise, prop_decrease=0.25) for i in range(len(data))]

# green_noise = AudioSegment.from_file("green_noise.m4a")
# sample_width = green_noise.sample_width
# frame_rate = green_noise.frame_rate
# green_noise = green_noise.set_channels(1).get_array_of_samples()
# green_noise = np.array([float(i) for i in green_noise])
# data_0 = np.array([float(i) for i in data[0]])
# filtered = nr.reduce_noise(data_0, green_noise)
# writer = wave.open("filtered.wav", mode="wb")
# writer.setnchannels(1)
# writer.setsampwidth(sample_width)
# writer.setframerate(frame_rate)
# writer.writeframes(filtered)
# writer.close()

print("Calculating stfts")
for item in metadata:
    item['path'] = "data/balanced/" + item["ytid"] + ".m4a"
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
times = [stft[1] for stft in stfts]

def chunks(lst, n):
    # https://stackoverflow.com/a/312464
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
hist_xvals = [np.array([np.mean(h, axis=0) for h in chunks(x, n)]) for x, n in zip(kt_xvals, binsize)]
hist_times = [np.array([h[0] for h in chunks(t, n)]) for t, n in zip(times, binsize)] # times where histogram starts
hist_X = np.concatenate(hist_xvals, axis=0)
num_segments_per_clip = [hist_xval.shape[0] for hist_xval in hist_xvals]

# let's try to cluster to label between the event happening and the event not happening
binarymeans = KMeans(n_clusters=2)
binarymeans.fit(hist_X)
bin_xvals = [binarymeans.predict(x) for x in hist_xvals]


def makevid(meta, labelling, h_times, length):
    outpath = 'results/' + meta['ytid'] + '.mp4'
    temppath = 'results/temp_' + meta['ytid'] + '.mp4'
    vidshape = (32,32)
    # Create a blank 300x300 black image
    red = np.zeros((vidshape[1], vidshape[0], 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    red[:] = (0, 0, 255)
    # Create a blank 300x300 black image
    green = np.zeros((vidshape[1], vidshape[0], 3), np.uint8)
    # Fill image with red color(set each pixel to green)
    green[:] = (0, 255, 0)

    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(temppath, fourcc, fps, vidshape, True)
    for i in range(int(length*fps)):
        time = i/fps
        idx = np.where(h_times <= time)[0][-1]
        if labelling[idx] == 1:
            video.write(green)
        else:
            video.write(red)
    cv2.destroyAllWindows()
    video.release()

    # audioclip = moviepy.editor.AudioFileClip(meta['path'])
    command = ['ffmpeg', '-i', temppath, '-i', meta['path'], '-c:v', 'copy', '-c:a', 'aac', outpath]
    res = subprocess.run(command, capture_output=True)
    os.remove(temppath)

# # makevid(metadata[i], bin_xvals[i], hist_times[i], times[i][-1])
# for i in range(len(metadata)):
#     makevid(metadata[i], bin_xvals[i], hist_times[i], times[i][-1])

# stop

# Train svm
test_Cs = [2**i for i in range(20, 0, -1)] # paper uses range(-7, 7)
test_gammas = [2**i for i in range(10, 0, -1)] # paper uses range(-7, 7)

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
    c, gamma = params
    clf = svm.SVC(C=c, kernel="rbf", gamma=gamma)
    clf.fit(X_train, y_train)
    return (c, gamma, clf)
svm_params = [(c, gamma) for c in test_Cs for gamma in test_gammas]
trained = Parallel(n_jobs=-1, verbose=50)(delayed(train_svm)(p) for p in svm_params)

def score_svm(params):
    c, gamma, clf = params
    error = 1 - clf.score(X_holdout, y_holdout)
    return (c, gamma, clf, error)
scored = Parallel(n_jobs=-1, verbose=50)(delayed(score_svm)(p) for p in trained)

min_idx = None
for i in range(len(scored)):
    if min_idx is None:
        min_idx = i
    elif scored[i][3] < scored[min_idx][3]:
        min_idx = i

min_err = scored[min_idx][3]

print("Minimum error:", min_err)
print("Minimum index:", min_idx)

best_C = scored[min_idx][0]
best_gamma = scored[min_idx][1]

print("Best C:", best_C)
print("Best gamma:", best_gamma)

scored_2d = []
for i in range(len(test_Cs)):
    temp = []
    for j in range(len(test_gammas)):
        this_c = test_Cs[i]
        this_gamma = test_gammas[j]
        for el in scored:
            c, gamma, clf, error = el
            if this_c == c and this_gamma == gamma:
                temp.append((c, gamma, clf, error))
                break
    scored_2d.append(temp)

prob_errs = [[el[3] for el in temp] for temp in scored_2d]

def plot_errs(prob_errs):
    plt.pcolormesh(np.array(prob_errs), shading="auto")
    plt.title("Holdout errors for varied parameters C and gamma")
    plt.ylabel("C")
    plt.yticks([i + 0.5 for i in range(len(test_Cs))], [test_Cs[i] for i in range(len(test_Cs))])
    plt.xlabel("Gamma")
    plt.xticks([i + 0.5 for i in range(len(test_gammas))], [test_gammas[i] for i in range(len(test_gammas))], rotation ='vertical')
    plt.show()

plot_errs(prob_errs)
