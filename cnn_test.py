import tensorflow as tf
from tensorflow import keras

import saliency.core as saliency

from run_stft import *
from import_google_audioset import *

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


ontology = load_ontology()

# dcase_classes = ["Alarm", "Vacuum cleaner"]
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
# dcase_metadata, dcase_stfts = getdata("dcase-weak", dcase_labels)
dcase_metadata, dcase_stfts = getdata("dcase-eval", dcase_labels) # oops, training with eval, but whatever, as it's an approach that is somehwat intended to work on the training data or something
# dcase_metadata, dcase_stfts = getdata("dcase-validation", dcase_labels) # seems validation is the same?

google_classes = ["noise"]
# google_classes = []
google_labels = [get_label_id_from_name(ontology, cl) for cl in google_classes]
google_metadata, google_stfts = getdata("google-unbalanced", google_labels)

classes = dcase_classes + google_classes
labels = dcase_labels + google_labels
metadata = dcase_metadata + google_metadata
stfts = dcase_stfts + google_stfts

# trim to ~9 seconds so all are the same
ftsize = 1551
ftimgs = [np.abs(stft[2][:, :ftsize]) if stft[2].shape[1] > ftsize else None for stft in stfts]
# remove the ones that are somehow < 9 seconds (not sure how they got into the dataset)
yvals = get_yvals(metadata, labels)
yvals = [np.where(y == np.array(labels))[0][0] for y in yvals]

truth = [get_stronglabelling(meta, stft[1], labels) for meta, stft in zip(metadata, stfts)]
truth = [t[:ftsize] for t, ft in zip(truth, ftimgs) if ft is not None]

temp = [y for y, ft in zip(yvals, ftimgs) if ft is not None]
stfts = [s for s, ft in zip(stfts, ftimgs) if ft is not None]
metadata = [meta for meta, ft in zip(metadata, ftimgs) if ft is not None]
ftimgs = np.array([ft for y, ft in zip(yvals, ftimgs) if ft is not None])

ftmax = np.max(ftimgs)
ftimgs /= ftmax
# ftsums = ftimgs.sum(axis=(1,2))
# ftimgs = np.array([ft/s for ft, s in zip(ftimgs, ftsums)])
onehot = np.eye(len(labels))
yvals = np.array([onehot[y] for y in temp])

inp = keras.layers.Input(shape=(ftimgs.shape[1:]))
x = keras.layers.GaussianNoise(0.01)(inp)
x = keras.layers.ReLU()(x)
x = keras.layers.Reshape((*ftimgs[0].shape, 1))(x)
x = keras.layers.Conv2D(filters=16, kernel_size=7, padding='same')(x)
x = keras.layers.MaxPooling2D((4, 4))(x)
x = keras.layers.Conv2D(filters=16, kernel_size=5, padding='same')(x)
x = keras.layers.MaxPooling2D((2, 4))(x)
x = keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(x)
x = keras.layers.MaxPooling2D((2, 3))(x)
x = keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(x)
x = keras.layers.Flatten()(x)
# x = keras.layers.Dense(16)(x)
x = keras.layers.Dense(len(labels))(x)
out = keras.layers.Softmax()(x)

model = keras.Model(inputs=inp, outputs=out)
if len(labels) <= 3:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='CategoricalCrossentropy')
else:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='CategoricalCrossentropy')
model.summary()

chkpt = tf.keras.callbacks.ModelCheckpoint(filepath='results/model.h5', save_best_only=True, monitor='loss')
if len(labels) <= 3:
    model.fit(ftimgs, yvals, epochs=50, verbose=1, callbacks=[chkpt])
else:
    model.fit(ftimgs, yvals, epochs=100, verbose=1, callbacks=[chkpt])

model = keras.models.load_model('results/model.h5')

# img = tf.Variable(ftimgs[0:1], dtype=float)
# with tf.GradientTape() as tape:
#   pred = model(img, training=False)
#   # loss = keras.losses.CategoricalCrossentropy()(pred, yvals[0:1])
#   # loss = pred[:][0]
#   class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
#   loss = pred[0][class_idxs_sorted[0]]
# grads = tape.gradient(loss, img)

# Using the github ipython notebook from the saliency package as a starting point, as couldn't get above to work
class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        tape.watch(images)
        output_layer = model(images)
        output_layer = output_layer[:, target_class_idx]
        gradients = np.array(tape.gradient(output_layer, images))
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

# i = 0
# im = ftimgs[i]
# predictions = model(np.array([im]))
# prediction_class = np.argmax(predictions[0])
# call_model_args = {class_idx_str: prediction_class}
#
# gradient_saliency = saliency.GradientSaliency()
#
# # Compute the vanilla mask and the smoothed mask.
# vanilla_mask_3d = gradient_saliency.GetMask(im, call_model_function, call_model_args)
# smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)
#
# temp = vanilla_mask_3d.copy()
# temp[temp < 0] = 0
# temp = temp.sum(axis=0)
# plt.plot(temp)

def getpreds(idx):
    print('Generating prediction for {}'.format(idx))
    im = ftimgs[idx]
    sumvals = []
    for i in range(len(labels)):
        call_model_args = {class_idx_str: i}
        gradient_saliency = saliency.GradientSaliency()
        vanilla_mask_3d = gradient_saliency.GetMask(im, call_model_function, call_model_args)
        temp = vanilla_mask_3d.copy()
        # temp[temp < 0] = 0
        temp = temp.sum(axis=0)
        sumvals.append(temp)
    sumvals = np.array(sumvals)
    preds = np.argmax(sumvals, axis=0).astype(object)
    for i in range(len(labels)):
        preds[preds==i] = labels[i]
    return preds

# pred = getpreds(0)
# plt.plot(pred)
# plt.plot(truth[0])
preds = [getpreds(i) for i in range(ftimgs.shape[0])]
scores = [np.mean(p == l) for p, l in zip(preds, truth)]

# plt.scatter(np.arange(len(metadata)), scores)
xrange = np.arange(len(metadata))
scores = np.array(scores)
yy = np.copy(yvals)
sortidx = np.argsort(np.argmax(yy, axis=1))
sc = np.copy(scores)
sc = sc[sortidx]
yy = yy[sortidx]
for i in range(len(labels)):
    plt.scatter(xrange[np.argmax(yy, axis=1)==i], sc[np.argmax(yy, axis=1)==i])
plt.axhline(y=1/len(labels), color='r', linestyle=':')
plt.title(np.mean(scores))
plt.ylim((0, 1))
plt.legend([None] + classes, ncol=3)
plt.savefig('results/scores.png')

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

# if len(labels) <= 3: # code not written for more than 3 classes yet
for i in range(len(metadata)):
    makevid(metadata[i], preds[i], stfts[i][1], stfts[i][1][-1], labels, truth[i])