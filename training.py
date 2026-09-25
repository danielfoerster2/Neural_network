#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import sys
import time

import matplotlib.pyplot    as plt
import numpy                as np
import pandas               as pd
import tensorflow           as tf

from sklearn.model_selection    import train_test_split



# ================================================================
# PARAMETERS
# ================================================================

EPOCHS = 50
PATIENCE = 10
N_PX = 128
N_SAMPLES = -1 # Set to -1 to use all samples in the dataset
PARAMETERS = ["n_atoms", "coreshell_index", "composition", "composition_Co"]
ARCHITECTURE = 3 # 1, 2 or 3



# ================================================================
# PATHS
# ================================================================

HRTEM_PATH = "/home/victor/Data/HRTEM_data/AgCo/"
SAVE_PATH = "/home/victor/Data/Predictions_data/"
if N_SAMPLES == -1:
    PREDICTIONS_FILE = SAVE_PATH + f"{EPOCHS}_epochs_{PATIENCE}_patience_all_samples_architecture_{ARCHITECTURE}_predictions.pkl"
else:
    PREDICTIONS_FILE = SAVE_PATH + f"{EPOCHS}_epochs_{PATIENCE}_patience_{N_SAMPLES}_samples_architecture_{ARCHITECTURE}_predictions.pkl"



# ================================================================
# FUNCTIONS
# ================================================================

def load_images(path, n_rows, n_cols, chunksize=10000):    
    ids = np.empty(n_rows, dtype=object)
    pixels = np.empty((n_rows, N_PX * N_PX), dtype=np.int8)
    dtype_map = {0: str}
    dtype_map.update({i: np.int8 for i in range(1, n_cols)})
    reader = pd.read_csv(path, delim_whitespace=True, header=None, dtype=dtype_map, na_filter=False, chunksize=chunksize, nrows=n_rows)# if n_samples !=-1 else None)
    pos = 0
    for chunk in reader:
        n = len(chunk)
        ids[pos:pos+n] = chunk.iloc[:, 0].to_numpy()
        pixels[pos:pos+n] = chunk.iloc[:, 1:].to_numpy(dtype=np.int8)
        pos += n
        del chunk
    return ids, pixels



def convolutional_neural_network():
    modele = tf.keras.models.Sequential()
    modele.add(tf.keras.layers.Input(shape=(N_PX, N_PX, 1)))
    for i in range(2):
        modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    if ARCHITECTURE == 2:
        modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    if ARCHITECTURE == 3:
        for i in range(2):
            modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
            modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
            modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    modele.add(tf.keras.layers.Flatten())
    modele.add(tf.keras.layers.Dense(1024, activation='relu'))
    modele.add(tf.keras.layers.Dense(768, activation='relu'))
    modele.add(tf.keras.layers.Dense(len(PARAMETERS), activation='linear'))
    return modele


def load_data():
    if N_SAMPLES == -1:
        df = pd.read_csv(HRTEM_PATH + "all_data.dat", delim_whitespace=True, na_values=["nan"], header=0)
    elif N_SAMPLES > 0:
        df = pd.read_csv(HRTEM_PATH + "all_data.dat", delim_whitespace=True, na_values=["nan"], header=0, nrows=N_SAMPLES)
    else:
        sys.exit("The number of samples must be greater than -1 to use all samples in the dataset (-1 for all).")
        
    #ids, pixels = load_images(HRTEM_PATH + "all_images.dat", df.shape[0], N_PX * N_PX + 1)
    
    n_rows = df.shape[0]
    ids = np.empty(n_rows, dtype=object)
    pixels = np.empty((n_rows, N_PX * N_PX), dtype=np.int8)
    dtype_map = {0: str}
    dtype_map.update({i: np.int8 for i in range(1, N_PX * N_PX + 1)})
    reader = pd.read_csv(HRTEM_PATH + "all_images.dat", delim_whitespace=True, header=None, dtype=dtype_map, na_filter=False, chunksize=10000, nrows=n_rows)# if n_samples !=-1 else None)
    pos = 0
    for chunk in reader:
        n = len(chunk)
        ids[pos:pos+n] = chunk.iloc[:, 0].to_numpy()
        pixels[pos:pos+n] = chunk.iloc[:, 1:].to_numpy(dtype=np.int8)
        pos += n
        del chunk

    imgs = (pixels + 128).astype(np.uint8).reshape(-1, N_PX, N_PX)
    del pixels
    df_img = pd.DataFrame({"id_sim": ids, "image": list(imgs)})
    del imgs
    del ids
    df = df.merge(df_img, on="id_sim", how="left")
    del df_img
    df['composition_Co'] = 1 - df['composition']
    df = df.dropna(subset=PARAMETERS)
    df = df.dropna(subset=["image"])
    return df




def train_model():
    df = load_data()
    training_set, test_set = train_test_split(df, train_size=0.6, random_state=42)
    val_size_adjusted = round(0.3 / (0.3 + 0.1), 2)
    validation_set, test_set = train_test_split(test_set, train_size=val_size_adjusted, random_state=42)

    data = {}
    for param in PARAMETERS:
        if param in df.columns:
            data[param] = {}
            y_train = training_set[param].to_numpy()
            data[param]["y_min"], data[param]["y_max"] = np.min(y_train), np.max(y_train)
            training_set[param] = (training_set[param] - data[param]["y_min"]) / (data[param]["y_max"] - data[param]["y_min"])
            validation_set[param] = (validation_set[param] - data[param]["y_min"]) / (data[param]["y_max"] - data[param]["y_min"])
            test_set[param] = (test_set[param] - data[param]["y_min"]) / (data[param]["y_max"] - data[param]["y_min"])
        else:
            sys.exit(f"Parameter '{param}' not found in the dataset.")

    X_train = np.stack(training_set["image"].to_numpy()).astype(np.float32)
    X_train = X_train[..., np.newaxis]
    y_train = training_set[PARAMETERS].to_numpy().astype(np.float32)
        
    X_val = np.stack(validation_set["image"].to_numpy()).astype(np.float32)
    X_val = X_val[..., np.newaxis]
    y_val = validation_set[PARAMETERS].to_numpy().astype(np.float32)
        
    X_test = np.stack(test_set["image"].to_numpy()).astype(np.float32)
    X_test = X_test[..., np.newaxis]
    y_test = test_set[PARAMETERS].to_numpy().astype(np.float32)

    datagen_training = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    # Revoir datagen: faire un datagen pour training et un autre pour validation et test, ou un pour training et validation et un autre pour test.

    training_gen = datagen_training.flow(X_train, y_train, batch_size=128, shuffle=True)
    validation_gen = datagen.flow(X_val, y_val, batch_size=128, shuffle=True)
    test_gen = datagen.flow(X_test, y_test, batch_size=128, shuffle=False)

    modele = convolutional_neural_network()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, PATIENCE//3), min_lr=1e-6)
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    modele.compile(optimizer=optimizer, loss='mse')
    history = modele.fit(training_gen, validation_data=validation_gen, epochs=EPOCHS, callbacks=callbacks, verbose=2) # Verbose=0 : No output, Verbose=1 : Progress bar, Verbose=2 : One line per epoch

    all_preds = modele.predict(test_gen)

    for i, param in enumerate(PARAMETERS):
        data[param]["y_true"] = test_set[param].to_numpy() * (data[param]["y_max"] - data[param]["y_min"]) + data[param]["y_min"]
        data[param]["y_pred"] = all_preds[:, i]
        data[param]["y_pred"] = data[param]["y_pred"] * (data[param]["y_max"] - data[param]["y_min"]) + data[param]["y_min"]
        data[param]['id_sim'] = test_set['id_sim'].to_numpy()
    
    data['performance'] = {}
    data['performance']['train_loss'] = history.history['loss']
    data['performance']['val_loss'] = history.history['val_loss']
    data['performance']['test_loss'] = history.history['val_loss']
    data['performance']['train_mse'] = modele.evaluate(training_gen)
    data['performance']['val_mse'] = modele.evaluate(validation_gen)
    data['performance']['test_mse'] = modele.evaluate(test_gen)

    return data



# ================================================================
# PLOTS
# ================================================================

def plot_training_loss(data, log_scale=False):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=.5)
    ax.plot(data['performance']['train_loss'], c="#4C72B0", lw=1.5, label='Training')
    ax.plot(data['performance']['val_loss'], c="#DD8452", lw=1.5, label='Validation')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
    if log_scale:
        ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    plt.tight_layout()
    os.makedirs("Figure", exist_ok=True)
    plt.savefig(f"Figure/training_loss.png", dpi=300)
    plt.close()


def plot_predictions_n_atoms(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=.5)
    ax.scatter(data["n_atoms"]["y_true"], data["n_atoms"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
    ax.plot([data["n_atoms"]["y_true"].min(), data["n_atoms"]["y_true"].max()], [data["n_atoms"]["y_true"].min(), data["n_atoms"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([data["n_atoms"]["y_true"].min(), 0]), np.max([data["n_atoms"]["y_true"].max(), 5000]))
    ax.set_ylim(np.min([data["n_atoms"]["y_pred"].min(), 0]), np.max([data["n_atoms"]["y_pred"].max(), 5000]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True number of atoms', fontsize=13)
    ax.set_ylabel('Predicted number of atoms', fontsize=13)
    plt.tight_layout()
    os.makedirs("Figure", exist_ok=True)
    plt.savefig(f"Figure/predictions_n_atoms.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_predictions_composition(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=.5)
    ax.scatter(data["composition"]["y_true"], data["composition"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
    ax.plot([data["composition"]["y_true"].min(), data["composition"]["y_true"].max()], [data["composition"]["y_true"].min(), data["composition"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([data["composition"]["y_true"].min(), 0]), np.max([data["composition"]["y_true"].max(), 1]))
    ax.set_ylim(np.min([data["composition"]["y_pred"].min(), 0]), np.max([data["composition"]["y_pred"].max(), 1]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True composition', fontsize=13)
    ax.set_ylabel('Predicted composition', fontsize=13)
    plt.tight_layout()
    os.makedirs("Figure", exist_ok=True)
    plt.savefig(f"Figure/predictions_composition.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_predictions_composition_Co(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=.5)
    ax.scatter(data["composition_Co"]["y_true"], data["composition_Co"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
    ax.plot([data["composition_Co"]["y_true"].min(), data["composition_Co"]["y_true"].max()], [data["composition_Co"]["y_true"].min(), data["composition_Co"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([data["composition_Co"]["y_true"].min(), 0]), np.max([data["composition_Co"]["y_true"].max(), 1]))
    ax.set_ylim(np.min([data["composition_Co"]["y_pred"].min(), 0]), np.max([data["composition_Co"]["y_pred"].max(), 1]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True composition', fontsize=13)
    ax.set_ylabel('Predicted composition', fontsize=13)
    plt.tight_layout()
    os.makedirs("Figure", exist_ok=True)
    plt.savefig(f"Figure/predictions_composition_Co.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions_coreshell_index(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=.5)
    ax.scatter(data["coreshell_index"]["y_true"], data["coreshell_index"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
    ax.plot([data["coreshell_index"]["y_true"].min(), data["coreshell_index"]["y_true"].max()], [data["coreshell_index"]["y_true"].min(), data["coreshell_index"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
    ax.set_xlim(np.min([data["coreshell_index"]["y_true"].min(), -0.5]), np.max([data["coreshell_index"]["y_true"].max(), 2]))
    ax.set_ylim(np.min([data["coreshell_index"]["y_pred"].min(), -0.5]), np.max([data["coreshell_index"]["y_pred"].max(), 2]))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
    ax.set_xlabel('True coreshell index', fontsize=13)
    ax.set_ylabel('Predicted coreshell index', fontsize=13)
    plt.tight_layout()
    os.makedirs("Figure", exist_ok=True)
    plt.savefig(f"Figure/predictions_coreshell_index.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_relative_error_n_atoms(data):
    n_bins = 50
    error = np.abs(data['n_atoms']['y_true'] - data['n_atoms']['y_pred'])
    relative_error = error / np.abs(data['n_atoms']['y_true']) * 100
    bins = np.linspace(data['n_atoms']['y_min'], data['n_atoms']['y_max'], n_bins + 1)
    #bins = np.linspace(100, data[label]['y_max'], n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_error = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (data['n_atoms']['y_true'] >= bins[i]) & (data['n_atoms']['y_true'] < bins[i+1])
        if np.sum(mask) > 0:
            mean_error[i] = np.mean(relative_error[mask])
        else:
            mean_error[i] = np.nan
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_axisbelow(True)
    ax.grid(alpha=.25, ls='-', lw=0.5)
    ax.bar(bin_centers, mean_error, width=bins[1] - bins[0], align='center', alpha=0.4, color='#5B9BD5', edgecolor='#2E5C8A', lw=0.6, label="Binned mean relative error")
    ax.plot(bin_centers, mean_error, c='#1B3A5C', marker='o', markersize=3.5, lw=1.3, label="Mean relative error")
    ax.axhline(np.nanmean(mean_error), color='#E91E63', ls='--', lw=1.5, label=f"Average error : {np.nanmean(mean_error):.2f}%")
    ax.set_ylabel("Mean relative error (%)", fontsize=13)
    #ax.set_xlim(0, 5000)
    ax.set_xlim(data['n_atoms']['y_min'], data['n_atoms']['y_max'])
    #ax.set_ylim(0, min(100, np.nanmax(mean_error) * 1.3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.tick_params(labelsize=11)
    ax.set_xlabel(f"True number of atoms", fontsize=13)
    ax.set_title(f"Prediction error vs. true number of atoms", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    os.makedirs(f"Figure", exist_ok=True)
    plt.savefig(f"Figure/relative_error_n_atoms.png", bbox_inches='tight', dpi=300)
    plt.close()

    

# ================================================================
# PROGRAM
# ================================================================

def main():
    start = time.time()
    update = False
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "rb") as f:
            data = pickle.load(f)
        for param in PARAMETERS:
            if param not in data.keys():
                data = train_model()
                update = True
                break
    else:
        data = train_model()
        update = True
    if update:
        os.makedirs(SAVE_PATH, exist_ok=True)
        with open(PREDICTIONS_FILE, "wb") as f:
            pickle.dump(data, f)
                           
    plot_training_loss(data)
    plot_predictions_n_atoms(data)
    plot_predictions_composition(data)
    plot_predictions_coreshell_index(data)
    plot_relative_error_n_atoms(data)

    print(f"Total time: {time.time() - start:.2f} seconds")

    
if __name__ == "__main__":
    main()