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

def getdata(splittype, labels):
    # Google
    if splittype.lower() == "google-balanced":
        metafile = "google_audioset_meta/balanced_train_segments.csv"
        folder = "data/google/balanced/"
    elif splittype.lower() == "google-eval":
        metafile = "google_audioset_meta/eval_segments.csv"
        folder = "data/google/eval/"
    elif splittype.lower() == "google-unbalanced":
        metafile = "google_audioset_meta/unbalanced_train_segments.csv"
        folder = "data/google/unbalanced/"
    # dcase
    elif splittype.lower() == "dcase-weak":
        metafile = "dcase_audioset_meta/training/weak.csv"
        folder = "data/dcase/weak/"
    elif splittype.lower() == "dcase-unlabel":
        metafile = "dcase_audioset_meta/training/unlabel_in_domain.csv"
        folder = "data/dcase/unlabel/"
    elif splittype.lower() == "dcase-eval":
        metafile = "dcase_audioset_meta/validation/eval_dcase2018.csv"
        folder = "data/dcase/eval/"
    elif splittype.lower() == "dcase-test":
        metafile = "dcase_audioset_meta/validation/test_dcase2018.csv"
        folder = "data/dcase/test/"
    elif splittype.lower() == "dcase-validation":
        metafile = "dcase_audioset_meta/validation/validation.csv"
        folder = "data/dcase/validation/"
    else:
        raise ValueError

    _meta = load_csv(metafile)
    meta = filter_labels(_meta, labels)
    audio = load_dataset(folder, meta)
    metadata = [audio[i][0] for i in range(len(audio))]
    data = [np.array(audio[i][1].set_channels(1).get_array_of_samples()) for i in range(len(audio))]
    freqs = [int(mediainfo(folder + item["ytid"] + ".m4a")["sample_rate"]) for item in metadata]
    for item in metadata:
        item["path"] = folder + item["ytid"] + ".m4a"
    stfts = [stft(data[i], fs=freqs[i], nperseg=512, nfft=1024) for i in range(len(data))]
    return metadata, stfts

def get_yvals(metadata, labels):
    y_labels = []
    for item in metadata:
        for label in labels:
            if 'strong_labels' in item:
                strlabels = [sl[0] for sl in item['strong_labels']]
                if label in strlabels:
                    y_labels.append(label)
                    break
            if 'weak_labels' in item:
                if label in item["weak_labels"]:
                    y_labels.append(label)
                    break # If multiple labels are assigned to the same item, just take the first label
    return y_labels

def get_stronglabelling(meta, htime, labels):
    # assume last label is noise
    noise = labels[-1]
    labels = labels[:-1]
    labelling = np.repeat(noise, len(htime)).astype(object)
    if 'strong_labels' not in meta:
        return labelling # assume we have noise class
    for sl in meta['strong_labels']:
        if sl[0] in labels:
            start = np.where(htime <= sl[1])[0][-1]
            stop = np.where(htime <= sl[2])[0][-1]
            labelling[start:stop] = sl[0]
    return labelling

def makevid(meta, labelling, h_times, length, labels, truth=None):
    outpath = 'results/' + meta['ytid'] + '.mp4'
    temppath = 'results/temp_' + meta['ytid'] + '.mp4'
    scalefactor = 10
    vidshape = (len(labels)*scalefactor, 2*scalefactor)
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(temppath, fourcc, fps, vidshape, True)
    for i in range(int(length * fps)):
        time = i / fps
        idx = np.where(h_times <= time)[0][-1]
        if idx >= len(labelling):
            break
        # img = np.zeros((vidshape[1]//scalefactor, vidshape[0]//scalefactor, 3), np.uint8)
        img = np.zeros((vidshape[1], vidshape[0], 3), np.uint8)
        # img[0, np.argwhere(np.array(labels)==labelling[idx])[0][0]] = (255, 255, 255)
        loc = np.argwhere(np.array(labels) == labelling[idx])[0][0]
        img[0:scalefactor, loc*scalefactor:(loc+1)*scalefactor] = (255, 255, 255)
        if truth is not None:
            # img[1, np.argwhere(np.array(labels)==truth[idx])[0][0]] = (255, 255, 255)
            loc = np.argwhere(np.array(labels)==truth[idx])[0][0]
            img[scalefactor:, loc*scalefactor:(loc+1)*scalefactor] = (255, 255, 255)
        # img = img.resize((vidshape[1], vidshape[0], 3))
        img = cv2.resize(img, vidshape)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

    # audioclip = moviepy.editor.AudioFileClip(meta['path'])
    command = ['ffmpeg', '-i', temppath, '-i', meta['path'], '-c:v', 'copy', '-c:a', 'aac', outpath]
    res = subprocess.run(command, capture_output=True)
    os.remove(temppath)

if __name__ == '__main__':
    ontology = load_ontology()

    # dcase_classes = ["Alarm", "Vacuum cleaner"]
    # dcase_classes = ["Vacuum cleaner"]
    dcase_classes = [
        "Alarm",
        "Speech",
        "Dog",
        "Cat",
        "Dishes, pots, and pans",
        "Frying (food)",
        "Electric toothbrush",
        "Vacuum cleaner",
        "Blender",
        "Water"]
    dcase_labels = [get_label_id_from_name(ontology, cl) for cl in dcase_classes]
    dcase_metadata, dcase_stfts = getdata("dcase-weak", dcase_labels)

    google_classes = ["noise"]
    # google_classes = []
    google_labels = [get_label_id_from_name(ontology, cl) for cl in google_classes]
    google_metadata, google_stfts = getdata("google-unbalanced", google_labels)

    classes = dcase_classes + google_classes
    labels = dcase_labels + google_labels
    metadata = dcase_metadata + google_metadata
    stfts = dcase_stfts + google_stfts

    # parameters:
    n_components = 100
    # timestep = 0.1
    # n_clusters = 7
    timestep = 1
    n_clusters = 6

    if len(classes) > 1:
        # dcase_val_metadata, dcase_val_stfts = getdata("dcase-validation", dcase_labels)
        # google_val_metadata, google_val_stfts = getdata("google-eval", google_labels)
        # val_metata = dcase_val_metadata + google_val_metadata
        # val_stfts = dcase_val_stfts + google_val_stfts
        val_metadata, val_stfts = getdata("dcase-validation", dcase_labels)

        google_eval_metadata, google_eval_stfts = getdata("google-eval", google_labels)
        dcase_eval_metadata, dcase_eval_stfts = getdata("dcase-eval", dcase_labels)
        eval_metadata = dcase_eval_metadata + google_eval_metadata
        eval_stfts = dcase_eval_stfts + google_eval_stfts

    y_labels = get_yvals(metadata, labels)

    def plot_stft(i):
        # plot the stft
        plt.pcolormesh(stfts[i][1], stfts[i][0], np.abs(stfts[i][2]), vmin=0, vmax=250, shading="auto")
        plt.title("STFT Magnitude of %s" % (metadata[i]["ytid"]))
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.savefig("results/stft_%s.png" % (metadata[i]["ytid"]), format="png")

    # for label in labels:
    #     plot_stft(y_labels.index(label))

    # Do PCA
    print("Running PCA")
    # n_components = 100
    pca = PCA(n_components=n_components)
    xvals = [np.abs(stft[2]).T for stft in stfts]
    # X = np.concatenate([np.abs(stft[2]) for stft in stfts], axis=1)
    X = np.concatenate(xvals, axis=0)
    pca.fit(X)
    t_xvals = [pca.transform(x) for x in xvals]
    t_X = pca.transform(X)

    def plot_pca(i):
        # plot the stft
        plt.pcolormesh(stfts[i][1], np.arange(0, n_components), t_xvals[i].transpose(), shading="auto")
        plt.title("PCA of %s" % (metadata[i]["ytid"]))
        plt.ylabel("Componets")
        plt.xlabel("Time [sec]")
        plt.savefig("results/pca_%s.png" % (metadata[i]["ytid"]), format="png")

    # for label in labels:
    #     plot_pca(y_labels.index(label))

    # Do K-Means
    print("Running K-means")
    # n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(t_X)
    onehot = np.eye(n_clusters)
    kt_xvals = [onehot[kmeans.predict(x)] for x in t_xvals]
    # kt_X = onehot[kmeans.predict(t_X)]

    def plot_clustering(i):
        # plot the clusters
        plt.pcolormesh(stfts[i][1], np.arange(0, n_clusters), kt_xvals[i].transpose(), shading="auto")
        plt.title("Clustering of %s" % (metadata[i]["ytid"]))
        plt.ylabel("Clusters")
        plt.xlabel("Time [sec]")
        plt.savefig("results/kmeans_%s.png" % (metadata[i]["ytid"]), format="png")

    # for label in labels:
    #     plot_clustering(y_labels.index(label))

    timediffs = [(stft[1][1] - stft[1][0]) for stft in stfts]
    # timestep = 0.1
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

    def plot_clustering_time_bins(i):
        # plot the clusters
        plt.pcolormesh(hist_times[i], np.arange(0, n_clusters), hist_xvals[i].transpose(), shading="auto")
        plt.title("Clustered time bins of %s" % (metadata[i]["ytid"]))
        plt.ylabel("Clusters")
        plt.xlabel("Time [sec]")
        plt.savefig("results/time_bin_%s.png" % (metadata[i]["ytid"]), format="png")

    # for label in labels:
    #     plot_clustering_time_bins(y_labels.index(label))

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

        def _makevid(meta, labelling, h_times, length):
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

        # _makevid(metadata[i], bin_xvals[i], hist_times[i], times[i][-1])
        for i in range(len(metadata)):
            _makevid(metadata[i], bin_xvals[i], hist_times[i], times[i][-1])

        # preds = bin_xvals
        # truth = [get_stronglabelling(meta, htime, labels) for meta, htime in zip(metadata, hist_times)]

    else:
        verbose = False
        # Train svm
        test_Cs = [2**i for i in range(6, -7, -1)] # paper uses range(-7, 7)
        test_gammas = [2**i for i in range(6, -7, -1)] # paper uses range(-7, 7)

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

        min_err = np.min(prob_errs)
        min_idx = np.unravel_index(prob_errs.argmin(), prob_errs.shape)

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

        val_vec, val_htimes = to_vectors(val_stfts)

        y_val = np.concatenate([get_stronglabelling(meta, htime, labels) for meta, htime in zip(val_metadata, val_htimes)])
        X_val = np.concatenate(val_vec)

        def fit_svm(k):
            # assume noise is last label
            class_weight = {}
            for l in labels[:-1]:
                class_weight[l] = 1
            class_weight[labels[-1]] = k
            clf = svm.SVC(C=best_C, kernel="rbf", gamma=best_gamma, class_weight=class_weight)#, probability=True)
            clf.fit(X_train, y_train)
            prob_err = 1 - clf.score(X_val, y_val)
            return clf, prob_err

        kvals = np.linspace(1, 3, 100)
        res = Parallel(n_jobs=-1, verbose=50)(delayed(fit_svm)(k) for k in kvals)
        errs = [r[1] for r in res]
        clf = res[np.argmin(errs)][0]

        eval_vec, eval_htimes = to_vectors(eval_stfts)
        eval_labelling = [get_stronglabelling(meta, htime, labels) for meta, htime in zip(eval_metadata, eval_htimes)]

        preds = [clf.predict(v) for v in eval_vec]

        # testres = [clf.predict(v) for v in hist_xvals]

        # def _makevid(meta, labelling, h_times, length, truth=None):
        #     outpath = 'results/' + meta['ytid'] + '.mp4'
        #     temppath = 'results/temp_' + meta['ytid'] + '.mp4'
        #     vidshape = (30, 30)
        #     fps = 30.0
        #     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #     video = cv2.VideoWriter(temppath, fourcc, fps, vidshape, True)
        #     for i in range(int(length*fps)):
        #         time = i/fps
        #         idx = np.where(h_times <= time)[0][-1]
        #         img = np.zeros((vidshape[1], vidshape[0], 3), np.uint8)
        #         if labelling[idx] == labels[0]:
        #             img[:, :10] = (255, 255, 255)
        #         elif labelling[idx] == labels[1]:
        #             img[:, 10:20] = (255, 255, 255)
        #         else:
        #             img[:, 20:] = (255, 255, 255)
        #         # vec = np.copy(dec_func[idx])
        #         # vec += np.min(vec)
        #         # vec /= np.linalg.norm(vec)
        #         # vec *= 255
        #         # vec = probs[idx]*255
        #         # img[15:, :10] = (vec[0], vec[0], vec[0])
        #         # img[15:, 10:20] = (vec[1], vec[1], vec[1])
        #         # img[15:, 20:] = (vec[2], vec[2], vec[2])
        #         if truth is not None:
        #             img[15:, :] = (0, 0, 0)
        #             if truth[idx] == labels[0]:
        #                 img[15:, :10] = (255, 255, 255)
        #             elif truth[idx] == labels[1]:
        #                 img[15:, 10:20] = (255, 255, 255)
        #             else:
        #                 img[15:, 20:] = (255, 255, 255)
        #         video.write(img)
        #     cv2.destroyAllWindows()
        #     video.release()
        #
        #     # audioclip = moviepy.editor.AudioFileClip(meta['path'])
        #     command = ['ffmpeg', '-i', temppath, '-i', meta['path'], '-c:v', 'copy', '-c:a', 'aac', outpath]
        #     res = subprocess.run(command, capture_output=True)
        #     os.remove(temppath)

        # _makevid(eval_metadata[0], preds[0], eval_htimes[0], eval_stfts[0][1][-1])
        for i in range(len(eval_metadata)):
            # _makevid(eval_metadata[i], probs[i], preds[i], eval_htimes[i], eval_stfts[i][1][-1])
            # _makevid(eval_metadata[i], preds[i], eval_htimes[i], eval_stfts[i][1][-1], eval_labelling[i])
            makevid(eval_metadata[i], preds[i], eval_htimes[i], eval_stfts[i][1][-1], labels, eval_labelling[i])

        # allnoise = np.repeat(labels[-1], len(np.concatenate(preds))).astype(object)
        # print(np.mean(np.concatenate(preds) == np.concatenate(eval_labelling)))
        # print(np.mean((allnoise) == np.concatenate(eval_labelling)))
        #
        scores = [np.mean(p == l) for p, l in zip(preds, eval_labelling)]
        # noisescores = [np.mean(labels[-1] == l) for l in eval_labelling]
        # label1scores = [np.mean(labels[0] == l) for l in eval_labelling]
        #
        # plt.plot(scores)
        # plt.plot(label1scores)

        yvals = get_yvals(eval_metadata, labels)
        yvals = [np.where(y == np.array(labels))[0][0] for y in yvals]
        onehot = np.eye(len(labels))
        yvals = np.array([onehot[y] for y in yvals])

        fig = plt.figure(figsize=[14, 7])
        xrange = np.arange(len(eval_metadata))
        scores = np.array(scores)
        yy = np.copy(yvals)
        sortidx = np.argsort(np.argmax(yy, axis=1))
        sc = np.copy(scores)
        sc = sc[sortidx]
        yy = yy[sortidx]
        for i in range(len(labels)):
            plt.scatter(xrange[np.argmax(yy, axis=1) == i], sc[np.argmax(yy, axis=1) == i], label='{}={:.3f}'.format(classes[i], np.mean(sc[np.argmax(yy, axis=1) == i])))
        plt.axhline(y=1 / len(labels), color='r', linestyle=':')
        plt.title('SVM Classification: {}'.format(np.mean(scores)))
        plt.ylim((0, 1))
        if len(labels) <= 3:
            plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.05))
        else:
            # plt.legend(ncol=3)
            plt.legend(bbox_to_anchor=(0.5,-0.009), loc="lower center", bbox_transform=fig.transFigure, ncol=6)
        plt.savefig('results/scores.png')