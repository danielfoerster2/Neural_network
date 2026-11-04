#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import time
import sys

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from train import RandomTranspose, Denormalize # registered once, so that "import train" is enough to load the model.keras of either script


##################
### PARAMETERS ###
##################

TRAINING_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1
EPOCHS = 30
PATIENCE = 10
N_PX = 256
BATCH_SIZE = 32 # STEM_DOWNSAMPLING = 2 needs about 4x the memory per image of the original ResNet50, for which 128 fits
PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
# True: start from the ImageNet weights. Compute nodes have no internet: download them once on a login node with
# python3 -c "import tensorflow as tf; tf.keras.applications.ResNet50(include_top=False, weights='imagenet')"
PRETRAINED = False
# Downsampling before the residual stages: 4 in the original ResNet50 (stride 2 convolution, then max pooling),
# 2 without the max pooling, which is the step that discards information; 1 without either
STEM_DOWNSAMPLING = 2
OUTPUT_DIR = "resnet50" # checkpoint, history, backup and results, apart from those of train.py

tf.keras.mixed_precision.set_global_policy("mixed_float16") # float16 on the V100 tensor cores, weights kept in float32



###############
### PROGRAM ###
###############

class Batches(tf.keras.utils.PyDataset):
    # Reads one batch at a time from the memory-mapped file; images stay uint8 until the Rescaling layer, on the GPU
    def __init__(self, images, y, idx, shuffle):
        super().__init__(workers=4)
        self.images, self.y, self.idx, self.shuffle = images, y, idx.copy(), shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / BATCH_SIZE))

    def __getitem__(self, i):
        idx = np.sort(self.idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        return self.images[idx][..., np.newaxis], self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def output(name):
    return os.path.join(OUTPUT_DIR, name)


def stem(layer):
    # Rebuilds each layer of the Keras ResNet50, except the max pooling of the stem (STEM_DOWNSAMPLING < 4) and its stride (STEM_DOWNSAMPLING = 1)
    if layer.name in ("pool1_pad", "pool1_pool") and STEM_DOWNSAMPLING < 4:
        return tf.keras.layers.Identity(name=layer.name)
    config = layer.get_config()
    if layer.name == "conv1_conv" and STEM_DOWNSAMPLING == 1:
        config["strides"] = (1, 1)
    return layer.__class__.from_config(config)


def build_model():
    inputs = tf.keras.layers.Input(shape=(N_PX, N_PX, 1), dtype="uint8")
    x = tf.keras.layers.Rescaling(1. if PRETRAINED else 1./255)(inputs) # grey levels 0-255 as in the ImageNet training, 0-1 otherwise
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x) # active during training only
    x = RandomTranspose()(x)
    if PRETRAINED:
        x = tf.keras.layers.Concatenate()([x, x, x]) # the ImageNet weights expect three colour channels
        x = tf.keras.layers.Normalization(mean=[103.939, 116.779, 123.68], variance=[1., 1., 1.])(x) # their preprocessing: mean of each channel (BGR) subtracted, no scaling
    # Stem, then stride 2 in the first block of stages 3 to 5: maps of N_PX / (8 * STEM_DOWNSAMPLING) pixels and 2048 features,
    # averaged so that the features are the same wherever the particle lies in the image
    backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet" if PRETRAINED else None,
                                              input_shape=(N_PX, N_PX, 3 if PRETRAINED else 1), pooling="avg")
    if STEM_DOWNSAMPLING < 4:
        weights = backbone.get_weights()
        backbone = tf.keras.models.clone_model(backbone, clone_function=stem)
        backbone.set_weights(weights) # the ImageNet ones if PRETRAINED: no weight depends on the strides
    x = backbone(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(PARAMETERS), activation='linear', dtype="float32")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_hrtem = np.load("all_data.npy", mmap_mode="r")
    meta = np.load("meta.npy")
    if len(meta) != len(data_hrtem):
        sys.exit(f"Error: meta.npy ({len(meta)} records) and all_data.npy ({len(data_hrtem)}) do not match; run load_data.py again")

    for param in PARAMETERS:
        if param not in meta.dtype.names:
            sys.exit(f"Error: Parameter '{param}' not found in the dataset.\nAvailable parameters are: {meta.dtype.names}")

    # The 10 images of one MD snapshot (id_sim = 10 * snapshot + rotation) go to the same set; same split as train.py
    groups = meta["id_sim"] // 10
    train_groups, test_groups = train_test_split(np.unique(groups), train_size=TRAINING_SIZE, random_state=42)
    val_groups, test_groups = train_test_split(test_groups, train_size=round(VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE), 2), random_state=42)
    train_idx, val_idx, test_idx = (np.flatnonzero(np.isin(groups, g)) for g in (train_groups, val_groups, test_groups))

    y = np.column_stack([meta[p] for p in PARAMETERS]).astype(np.float32)
    y_fit = np.column_stack([np.cbrt(y[:, i]) if p == "n_atoms" else y[:, i] for i, p in enumerate(PARAMETERS)]) # N^(1/3) scales like the particle size in the image
    y_min = np.min(y_fit[train_idx], axis=0)
    y_max = np.max(y_fit[train_idx], axis=0)
    y_norm = (y_fit - y_min) / (y_max - y_min)

    images = data_hrtem["image"] # still memory-mapped, uint8
    training_gen = Batches(images, y_norm, train_idx, shuffle=True)
    train_eval_gen = Batches(images, y_norm, train_idx, shuffle=False)
    validation_gen = Batches(images, y_norm, val_idx, shuffle=False)
    test_gen = Batches(images, y_norm, test_idx, shuffle=False)

    print(f"Time to prepare data: {time.time() - start_time:.2f} seconds")

    model = build_model()
    # A job stopped before the end (time limit) leaves OUTPUT_DIR/backup/: submitting it again resumes after the last completed epoch
    resume = os.path.isdir(output("backup"))
    best_val_loss, initial_epoch = None, 0
    if resume and os.path.exists(output("history.csv")):
        previous = np.atleast_1d(np.genfromtxt(output("history.csv"), delimiter=",", names=True))
        best_val_loss, initial_epoch = float(np.min(previous["val_loss"])), len(previous) # Keras 3.0 does not restore the epoch counter itself
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=1e-4),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE//3, min_delta=1e-6),
        tf.keras.callbacks.ModelCheckpoint(output("checkpoint.keras"), monitor='val_loss', save_best_only=True, initial_value_threshold=best_val_loss), # best epoch over all jobs (normalised outputs)
        tf.keras.callbacks.CSVLogger(output("history.csv"), append=resume), # losses of every epoch, over all jobs
        tf.keras.callbacks.BackupAndRestore(output("backup")) # last, so that a backup marks a fully completed epoch; deleted when training ends
    ]
    # Keras defaults (weight decay 0.004, no gradient clipping); smaller steps when fine-tuning, so as not to wipe out the ImageNet features
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4 if PRETRAINED else 1e-3)

    model.compile(optimizer=optimizer, loss='mse')
    model.optimizer.build(model.trainable_variables) # built before BackupAndRestore reloads the backup, so the optimizer state is reloaded too (Keras 3.0 builds it later)
    model.fit(training_gen, validation_data=validation_gen, epochs=EPOCHS, initial_epoch=initial_epoch, callbacks=callbacks, verbose=2)
    model.load_weights(output("checkpoint.keras")) # best epoch over all jobs (EarlyStopping only sees the epochs of this job)
    history = np.atleast_1d(np.genfromtxt(output("history.csv"), delimiter=",", names=True))
    # Weights of the best epoch. Input: uint8 images (n, N_PX, N_PX, 1); output: PARAMETERS in physical units.
    # To load it elsewhere: import train (registers the custom layers), then tf.keras.models.load_model("resnet50/model.keras")
    predictor = tf.keras.Model(model.inputs, Denormalize(y_min, y_max, [p == "n_atoms" for p in PARAMETERS], dtype="float32")(model.outputs[0]))
    predictor.save(output("model.keras"))
    predictions = predictor.predict(test_gen, verbose=0)

    data = {}
    for i, param in enumerate(PARAMETERS):
        data[param] = {
            "y_true" : y[test_idx, i],
            "y_pred" : predictions[:, i]
        }

    data["performance"] = {
        "loss": history['loss'],
        "val_loss": history['val_loss'],
        "train_mse": model.evaluate(train_eval_gen, verbose=0),
        "val_mse": model.evaluate(validation_gen, verbose=0),
        "test_mse": model.evaluate(test_gen, verbose=0)
    }

    with open(output("predictions.pkl"), "wb") as f:
        pickle.dump(data, f)


    print(f"Total time: {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    main()
