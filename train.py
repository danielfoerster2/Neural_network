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


##################
### PARAMETERS ###
##################

TRAINING_SIZE = 0.6
VALIDATION_SIZE = 0.3
TEST_SIZE = 0.1
EPOCHS = 50
PATIENCE = 10
N_PX = 256
BATCH_SIZE = 128
PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
SAVE_PATH = "/home/victor/Data/Predictions_data/avec_substrat/"

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


def main():
    start_time = time.time()
    
    data_hrtem = np.load("all_data.npy", mmap_mode="r")
    
    for param in PARAMETERS:
        if param not in data_hrtem.dtype.names:
            sys.exit(f"Error: Parameter '{param}' not found in the dataset.\nAvailable parameters are: {data_hrtem.dtype.names}")
            
    # The 10 images of one MD snapshot (id_sim = 10 * snapshot + rotation) go to the same set
    groups = data_hrtem["id_sim"] // 10
    train_groups, test_groups = train_test_split(np.unique(groups), train_size=TRAINING_SIZE, random_state=42)
    val_groups, test_groups = train_test_split(test_groups, train_size=round(VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE), 2), random_state=42)
    train_idx, val_idx, test_idx = (np.flatnonzero(np.isin(groups, g)) for g in (train_groups, val_groups, test_groups))

    y = np.column_stack([data_hrtem[p] for p in PARAMETERS]).astype(np.float32)
    y_min = np.min(y[train_idx], axis=0)
    y_max = np.max(y[train_idx], axis=0)
    y_norm = (y - y_min) / (y_max - y_min)

    images = data_hrtem["image"] # still memory-mapped, uint8
    training_gen = Batches(images, y_norm, train_idx, shuffle=True)
    train_eval_gen = Batches(images, y_norm, train_idx, shuffle=False)
    validation_gen = Batches(images, y_norm, val_idx, shuffle=False)
    test_gen = Batches(images, y_norm, test_idx, shuffle=False)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_PX, N_PX, 1), dtype="uint8"),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"), # active during training only
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(768, activation='relu'),
        tf.keras.layers.Dense(len(PARAMETERS), activation='linear', dtype="float32")
    ])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE//3, min_delta=1e-6)
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(training_gen, validation_data=validation_gen, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    predictions = model.predict(test_gen, verbose=0)
    
    data = {}    
    for i, param in enumerate(PARAMETERS):
        data[param] = {
            "y_true" : y[test_idx, i],
            "y_pred" : predictions[:, i] * (y_max[i] - y_min[i]) + y_min[i]
        }
    
    data["performance"] = {
        "loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        "train_mse": model.evaluate(train_eval_gen, verbose=0),
        "val_mse": model.evaluate(validation_gen, verbose=0),
        "test_mse": model.evaluate(test_gen, verbose=0)
    }
    
    with open(os.path.join("predictions.pkl"), "wb") as f:
        pickle.dump(data, f)
    
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    main()