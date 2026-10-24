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
PARAMETERS = ["n_atoms", "composition", "coreshell_index"]
SAVE_PATH = "/home/victor/Data/Predictions_data/avec_substrat/"



###############
### PROGRAM ###
###############

def main():
    start_time = time.time()
    
    data_hrtem = np.load("all_data.npy", mmap_mode="r")
    
    for param in PARAMETERS:
        if param not in data_hrtem.dtype.names:
            sys.exit(f"Error: Parameter '{param}' not found in the dataset.\nAvailable parameters are: {data_hrtem.dtype.names}")
            
    training_set, test_set = train_test_split(data_hrtem, train_size=TRAINING_SIZE, random_state=42)
    validation_set, test_set = train_test_split(test_set, train_size=round(VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE), 2), random_state=42)

    X_train = np.stack(training_set["image"]).astype(np.float32)
    X_train = X_train[..., np.newaxis]
    y_train = np.column_stack([training_set[p] for p in PARAMETERS]).astype(np.float32)

    X_val = np.stack(validation_set["image"]).astype(np.float32)
    X_val = X_val[..., np.newaxis]
    y_val = np.column_stack([validation_set[p] for p in PARAMETERS]).astype(np.float32)
        
    X_test = np.stack(test_set["image"]).astype(np.float32)
    X_test = X_test[..., np.newaxis]
    y_test = np.column_stack([test_set[p] for p in PARAMETERS]).astype(np.float32)

    y_min = np.min(y_train, axis=0)
    y_max = np.max(y_train, axis=0)

    y_train = (y_train - y_min) / (y_max - y_min)
    y_val = (y_val - y_min) / (y_max - y_min)
    y_test = (y_test - y_min) / (y_max - y_min)    

    datagen_training = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    
    training_gen = datagen_training.flow(X_train, y_train, batch_size=128, shuffle=True)
    train_eval_gen = datagen.flow(X_train, y_train, batch_size=128, shuffle=False)
    #validation_gen = datagen.flow(X_val, y_val, batch_size=128, shuffle=True)
    validation_gen = datagen.flow(X_val, y_val, batch_size=128, shuffle=False)
    test_gen = datagen.flow(X_test, y_test, batch_size=128, shuffle=False)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_PX, N_PX, 1)),
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
        tf.keras.layers.Dense(len(PARAMETERS), activation='linear')
    ])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE//3, min_delta=1e-6)
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(training_gen, validation_data=validation_gen, epochs=EPOCHS, callbacks=callbacks)
    predictions = model.predict(test_gen)
    
    data = {}    
    for i, param in enumerate(PARAMETERS):
        data[param] = {
            "y_true" : test_set[param],
            "y_pred" : predictions[:, i] * (y_max[i] - y_min[i]) + y_min[i]
            #"y_min" : y_min[i],
            #"y_max" : y_max[i]
        }
    
    data["performance"] = {
        "loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        #"train_mse": model.evaluate(training_gen),
        "train_mse": model.evaluate(train_eval_gen),
        "val_mse": model.evaluate(validation_gen),
        "test_mse": model.evaluate(test_gen)
    }
    
    #with open(os.path.join(SAVE_PATH, "predictions.pkl"), "wb") as f:
    with open(os.path.join("predictions.pkl"), "wb") as f:
        pickle.dump(data, f)
    
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    main()