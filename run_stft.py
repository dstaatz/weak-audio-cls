#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import stft
# from scipy.linalg import svd

from import_google_audioset import *
from pydub.utils import mediainfo

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm

import cv2
import subprocess

eval_meta = load_csv("google_audioset_meta/eval_segments.csv")
balanced_meta = load_csv("google_audioset_meta/balanced_train_segments.csv")
unbalanced_meta = load_csv("google_audioset_meta/unbalanced_train_segments.csv")
ontology = load_ontology()

# Load in a dataset and perform feature extraction
classes = ['hammer', 'drill', 'noise']
# classes = ['hammer']
labels = [get_label_id_from_name(ontology, cl) for cl in classes]

def getdata(splittype):
    if splittype == 'balanced':
        _meta = balanced_meta
        folder = 'data/balanced/'
    elif splittype == 'eval':
        _meta = eval_meta
        folder = 'data/eval/'
    else:
        _meta = unbalanced_meta
        folder = 'data/unbalanced/'
    meta = filter_labels(_meta, labels)
    audio = load_dataset(folder, meta)
    metadata = [audio[i][0] for i in range(len(audio))]
    data = [np.array(audio[i][1].set_channels(1).get_array_of_samples()) for i in range(len(audio))]
    freqs = [int(mediainfo(folder + item["ytid"] + ".m4a")['sample_rate']) for item in metadata]
    for item in metadata:
        item['path'] = folder + item["ytid"] + ".m4a"
    stfts = [stft(data[i], fs=freqs[i], nperseg=512, nfft=1024) for i in range(len(data))]
    return metadata, stfts

metadata, stfts = getdata('balanced')

def get_yvals(metadata):
    y_labels = []
    for item in metadata:
        for label in labels:
            if label in item["positive_labels"]:
                y_labels.append(label)
                break # If multiple labels are assigned to the same item, just take the first label
    return y_labels

y_labels = get_yvals(metadata)

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

def to_vectors(stfts):
    times = [stft[1] for stft in stfts]
    xvals = [np.abs(stft[2]).T for stft in stfts]
    t_xvals = [pca.transform(x) for x in xvals]
    kt_xvals = [onehot[kmeans.predict(x)] for x in t_xvals]
    timediffs = [(stft[1][1] - stft[1][0]) for stft in stfts]
    binsize = [int(timestep/td) for td in timediffs]
    hist_xvals = [np.array([np.mean(h, axis=0) for h in chunks(x, n)]) for x, n in zip(kt_xvals, binsize)]
    hist_times = [np.array([h[0] for h in chunks(t, n)]) for t, n in zip(times, binsize)]  # times where histogram starts
    return hist_xvals, hist_times

if len(classes) == 1:
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

    # makevid(metadata[i], bin_xvals[i], hist_times[i], times[i][-1])
    for i in range(len(metadata)):
        makevid(metadata[i], bin_xvals[i], hist_times[i], times[i][-1])

else:
    verbose = False
    # Train svm
    test_Cs = [2**i for i in range(-7, 10)] # paper uses range(-7, 7)
    test_gammas = [2**i for i in range(-7, 10)] # paper uses range(-7, 7)

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
    # clfs = []
    # for C in test_Cs:
    #     temp = []
    #     for gamma in test_gammas:
    #         if verbose:
    #             print("Training SVM (C = %s, gamma = %s)" % (str(C), str(gamma)))
    #         clf = svm.SVC(C=C, kernel="rbf", gamma=gamma)
    #         clf.fit(X_train, y_train)
    #         temp.append(clf)
    #     clfs.append(temp)
    params = []
    for C in test_Cs:
        for gamma in test_gammas:
            params.append((verbose, C, gamma))

    def fit_svm(p):
        verbose, C, gamma = p
        if verbose:
            print("Training SVM (C = %s, gamma = %s)" % (str(C), str(gamma)))
        clf = svm.SVC(C=C, kernel="rbf", gamma=gamma)
        clf.fit(X_train, y_train)
        if verbose:
            print("Error (C = %s, gamma = %s): " % (str(test_Cs[i]), str(test_gammas[j])), end="")
        prob_err = 1 - clf.score(X_holdout, y_holdout)
        if verbose:
            print(prob_err)
        return clf, prob_err

    res = Parallel(n_jobs=-1, verbose=50)(delayed(fit_svm)(p) for p in params)
    clfs = [r[0] for r in res]
    prob_errs = [r[1] for r in res]
    clfs = np.array(clfs).reshape((len(test_Cs), len(test_gammas)))
    prob_errs = np.array(prob_errs).reshape((len(test_Cs), len(test_gammas)))
    if verbose:
        print()

    # Determine the best C and gamma for a radial basis kernel
    # prob_errs = []
    # for i in range(len(test_Cs)):
    #     temp = []
    #     for j in range(len(test_gammas)):
    #         if verbose:
    #             print("Error (C = %s, gamma = %s): " % (str(test_Cs[i]), str(test_gammas[j])), end="")
    #         prob_err = 1 - clfs[i][j].score(X_holdout, y_holdout)
    #         if verbose:
    #             print(prob_err)
    #         temp.append(prob_err)
    #     prob_errs.append(temp)

    # min_err = min([min(temp) for temp in prob_errs])
    min_err = np.min(prob_errs)
    min_idx = np.unravel_index(prob_errs.argmin(), prob_errs.shape)

    # min_idx = None
    # for (i, temp) in enumerate(prob_errs):
    #     if min_err in temp:
    #         min_idx = (i, temp.index(min_err))
    #         break

    if verbose:
        print()
        print("Minimum error:", min_err)
        print("Minimum index:", min_idx)

    best_C = test_Cs[min_idx[0]]
    best_gamma = test_gammas[min_idx[1]]

    if verbose:
        print("Best C:", best_C)
        print("Best gamma:", best_gamma)

    def plot_errs(prob_errs):
        plt.pcolormesh(np.array(prob_errs), shading="auto")
        plt.title("Holdout errors for varied parameters C and gamma")
        plt.ylabel("C")
        plt.yticks([i + 0.5 for i in range(len(test_Cs))], [test_Cs[i] for i in range(len(test_Cs))])
        plt.xlabel("Gamma")
        plt.xticks([i + 0.5 for i in range(len(test_Cs))], [test_Cs[i] for i in range(len(test_Cs))], rotation ='vertical')
        plt.show()

    if verbose:
        plot_errs(prob_errs)

    y_train = np.concatenate([[y_labels[idx] for i in range(num_segments_per_clip[idx])] for idx in range(len(hist_xvals))])
    X_train = np.concatenate(hist_xvals)
    clf = svm.SVC(C=best_C, kernel="rbf", gamma=best_gamma, class_weight={labels[0]: 1, labels[1]: 1, labels[2]: 2})#, probability=True)
    clf.fit(X_train, y_train)

    eval_metadata, eval_stfts = getdata('eval')
    eval_vec, eval_htimes = to_vectors(eval_stfts)
    preds = [clf.predict(v) for v in eval_vec]
    # probs = [clf.predict_proba(v) for v in eval_vec]
    # dec_funcs = [clf.decision_function(v) for v in eval_vec]

    testres = [clf.predict(v) for v in hist_xvals]

    def makevid(meta, labelling, h_times, length):
        outpath = 'results/' + meta['ytid'] + '.mp4'
        temppath = 'results/temp_' + meta['ytid'] + '.mp4'
        vidshape = (30, 30)
        fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(temppath, fourcc, fps, vidshape, True)
        for i in range(int(length*fps)):
            time = i/fps
            idx = np.where(h_times <= time)[0][-1]
            img = np.zeros((vidshape[1], vidshape[0], 3), np.uint8)
            if labelling[idx] == labels[0]:
                img[:, :10] = (255, 255, 255)
            elif labelling[idx] == labels[1]:
                img[:, 10:20] = (255, 255, 255)
            else:
                img[:, 20:] = (255, 255, 255)
            # vec = np.copy(dec_func[idx])
            # vec += np.min(vec)
            # vec /= np.linalg.norm(vec)
            # vec *= 255
            # vec = probs[idx]*255
            # img[15:, :10] = (vec[0], vec[0], vec[0])
            # img[15:, 10:20] = (vec[1], vec[1], vec[1])
            # img[15:, 20:] = (vec[2], vec[2], vec[2])
            video.write(img)
        cv2.destroyAllWindows()
        video.release()

        # audioclip = moviepy.editor.AudioFileClip(meta['path'])
        command = ['ffmpeg', '-i', temppath, '-i', meta['path'], '-c:v', 'copy', '-c:a', 'aac', outpath]
        res = subprocess.run(command, capture_output=True)
        os.remove(temppath)

    # makevid(eval_metadata[0], preds[0], eval_htimes[0], eval_stfts[0][1][-1])
    for i in range(len(eval_metadata)):
        # makevid(eval_metadata[i], probs[i], preds[i], eval_htimes[i], eval_stfts[i][1][-1])
        makevid(eval_metadata[i], preds[i], eval_htimes[i], eval_stfts[i][1][-1])