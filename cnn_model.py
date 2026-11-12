"""
PHASE 4: Deep Learning Model with CNN (Segment-Based)
======================================================
Splits each 30-second track into 10 segments of 3 seconds each.
Each segment → Mel-Spectrogram → CNN input.

Why segment-based?
  - GTZAN has 1000 tracks → 10,000 segments: 10x more training data
  - At test time: majority vote over all segment predictions per track
  - Data leakage prevention: train/test split is done at the TRACK level

Architecture:
  3-block CNN (Conv→BN→Conv→BN→Pool→Dropout) + GlobalAvgPool + Dense
  Per segment: (64 × 64 × 1) mel-spectrogram
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

GENRES      = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']
IMG_SIZE    = 64       # Spectrogram size per segment
N_MELS      = 64
SEG_SECONDS = 3        # Seconds per segment
DURATION    = 30       # Total audio duration
SR          = 22050
EPOCHS      = 60
BATCH       = 64



def segment_to_melspec(segment, sr, img_size=IMG_SIZE):
    """Converts an audio segment to a normalized (img_size x img_size x 1) mel-spectrogram."""
    mel    = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS,
                                            hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_rs = tf.image.resize(mel_db[..., np.newaxis], [img_size, img_size]).numpy()
    mel_n  = (mel_rs - mel_rs.min()) / (mel_rs.max() - mel_rs.min() + 1e-8)
    return mel_n.astype(np.float32)



def build_segment_dataset(audio_root):
    """
    Splits each track into SEG_SECONDS-long segments.
    Returns:
      X        : (N_seg, IMG_SIZE, IMG_SIZE, 1)  — mel-spectrograms
      y        : (N_seg,)                        — genre labels (string)
      song_ids : (N_seg,)                        — which track each segment belongs to
    """
    cache_X   = os.path.join(DATA_DIR, 'cnn_X_seg.npy')
    cache_y   = os.path.join(DATA_DIR, 'cnn_y_seg.npy')
    cache_ids = os.path.join(DATA_DIR, 'cnn_ids_seg.npy')

    if os.path.exists(cache_X):
        print("Loading segments from cache...")
        X   = np.load(cache_X)
        y   = np.load(cache_y, allow_pickle=True)
        ids = np.load(cache_ids, allow_pickle=True)
        print(f"X: {X.shape}, unique songs: {len(np.unique(ids))}")
        return X, y, ids

    seg_samples = int(SR * SEG_SECONDS)
    total_load  = int(SR * DURATION)

    X, y, ids = [], [], []
    song_counter = 0

    for genre in GENRES:
        genre_path = os.path.join(audio_root, genre)
        if not os.path.exists(genre_path):
            print(f"  WARNING: {genre_path} not found, skipping.")
            continue

        files = sorted([f for f in os.listdir(genre_path) if f.endswith(('.wav', '.au'))])
        print(f"[{genre.upper()}] {len(files)} tracks × ~{DURATION//SEG_SECONDS} segments...")

        for fname in files:
            try:
                audio, sr = librosa.load(
                    os.path.join(genre_path, fname),
                    duration=DURATION, mono=True, sr=SR)

                # Non-overlapping 3-second segment windows
                for start in range(0, len(audio) - seg_samples + 1, seg_samples):
                    seg  = audio[start : start + seg_samples]
                    spec = segment_to_melspec(seg, sr)
                    X.append(spec)
                    y.append(genre)
                    ids.append(f"{genre}/{fname}")

                song_counter += 1
            except Exception as e:
                print(f"  ERROR: {fname} -> {e}")

    X   = np.array(X,   dtype=np.float32)
    y   = np.array(y)
    ids = np.array(ids)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(cache_X,   X)
    np.save(cache_y,   y)
    np.save(cache_ids, ids)
    print(f"\nSaved to cache.")
    print(f"Total: {len(X)} segments, {song_counter} tracks")
    print(f"Segments per track: ~{len(X)//song_counter}")
    return X, y, ids


# ---------------------------------------------------------------------------
# Train / Test split — at TRACK level
# ---------------------------------------------------------------------------
def song_level_split(y_labels, song_ids, test_size=0.2, random_state=42):
    """
    Splits at the track ID level to prevent data leakage.
    Segments from the same track never appear in both train and test sets.
    """
    unique_songs = np.unique(song_ids)
    song_genres  = [y_labels[song_ids == s][0] for s in unique_songs]

    train_songs, test_songs = train_test_split(
        unique_songs, test_size=test_size,
        random_state=random_state, stratify=song_genres)

    train_mask = np.isin(song_ids, train_songs)
    test_mask  = np.isin(song_ids, test_songs)
    return train_mask, test_mask, test_songs


# ---------------------------------------------------------------------------
# CNN Architecture — clean design with strong regularization
# ---------------------------------------------------------------------------
def build_cnn(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    """
    3-block CNN:
      Each block: Conv→BN→Conv→BN→MaxPool→Dropout
      Head: GlobalAveragePooling → Dense(256) → Dropout → Softmax
    GlobalAvgPool has fewer parameters than Flatten → less overfitting.
    """
    reg = regularizers.l2(1e-4)

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      kernel_regularizer=reg, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.30),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(num_classes, activation='softmax'),
    ], name='music_cnn')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# Song-level majority vote evaluation
# ---------------------------------------------------------------------------
def song_level_accuracy(model, X_test, y_test_enc, test_song_ids, le):
    """
    Runs all segments of each test track through the model,
    averages the softmax outputs and predicts the track genre.
    """
    seg_probs = model.predict(X_test, verbose=0)   # (N_seg, 10)

    unique_test_songs = np.unique(test_song_ids)
    true_labels, pred_labels = [], []

    for song in unique_test_songs:
        mask       = test_song_ids == song
        avg_probs  = seg_probs[mask].mean(axis=0)   # average over segments
        pred_class = np.argmax(avg_probs)
        true_class = y_test_enc[mask][0]

        pred_labels.append(pred_class)
        true_labels.append(true_class)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    acc = accuracy_score(true_labels, pred_labels)

    print(f"\nSong-level Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("\nClassification Report (song level):")
    print(classification_report(true_labels, pred_labels,
                                target_names=le.classes_))
    return true_labels, pred_labels, acc


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title, ylabel in zip(
        axes,
        ['accuracy', 'loss'],
        ['CNN — Accuracy (Segment)', 'CNN — Loss (Segment)'],
        ['Accuracy', 'Loss']
    ):
        ax.plot(history.history[key],          label='Train',      color='steelblue')
        ax.plot(history.history[f'val_{key}'], label='Validation', color='coral')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['CNN — Confusion Matrix (Song Level, Count)',
         'CNN — Confusion Matrix (Song Level, Normalized)'],
        ['d', '.2f']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual',    fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sample_spectrograms(X, y_labels, le, save_path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Mel-Spectrogram Samples — 3-second segments',
                 fontsize=14, fontweight='bold')
    for idx, genre in enumerate(le.classes_):
        ax = axes[idx // 5][idx % 5]
        genre_idx = np.where(y_labels == genre)[0][0]
        im = ax.imshow(X[genre_idx, :, :, 0], aspect='auto',
                       origin='lower', cmap='magma')
        ax.set_title(genre.capitalize(), fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, format='%.2f')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main(audio_root=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    if audio_root is None:
        import kagglehub
        from feature_extraction import find_audio_root
        path       = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification")
        audio_root = find_audio_root(path)

    print(f"\nAudio root: {audio_root}")

    # Build dataset
    print("\nPreparing segment-based dataset...")
    X, y_labels, song_ids = build_segment_dataset(audio_root)

    le    = LabelEncoder().fit(GENRES)
    y_enc = le.transform(y_labels)

    # Sample spectrogram visualization
    plot_sample_spectrograms(
        X, y_labels, le,
        os.path.join(RESULTS_DIR, 'sample_spectrograms.png'))

    # Train / Test split (track level)
    train_mask, test_mask, test_songs = song_level_split(y_labels, song_ids)

    X_train, y_train = X[train_mask], y_enc[train_mask]
    X_test,  y_test  = X[test_mask],  y_enc[test_mask]
    test_song_ids    = song_ids[test_mask]

    print(f"\nTraining: {X_train.shape[0]} segments  |  Test: {X_test.shape[0]} segments")
    print(f"Number of test tracks: {len(test_songs)}")

    # Build model
    model = build_cnn(num_classes=len(GENRES))
    model.summary(line_length=80)

    # Callbacks
    ckpt_path = os.path.join(MODELS_DIR, 'cnn_best.keras')
    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=12,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                                  save_best_only=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=6, min_lr=1e-6, verbose=1),
    ]

    # Shuffle before validation_split so all genres are represented
    shuffle_idx = np.random.default_rng(42).permutation(len(X_train))
    X_train_sh  = X_train[shuffle_idx]
    y_train_sh  = y_train[shuffle_idx]

    print(f"\nStarting CNN training ({EPOCHS} epochs, batch={BATCH})...")
    history = model.fit(
        X_train_sh, y_train_sh,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=0.15,
        callbacks=cb,
        verbose=1
    )

    # Segment-level evaluation
    seg_loss, seg_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"Segment-level Test Accuracy : {seg_acc:.4f}  ({seg_acc*100:.2f}%)")

    # Song-level evaluation (majority vote)
    true_labels, pred_labels, song_acc = song_level_accuracy(
        model, X_test, y_test, test_song_ids, le)

    # Plots
    plot_training_history(
        history,
        os.path.join(RESULTS_DIR, 'cnn_training_history.png'))
    plot_confusion_matrix(
        true_labels, pred_labels, le.classes_,
        os.path.join(RESULTS_DIR, 'cm_cnn.png'))

    # Save
    joblib.dump(le, os.path.join(MODELS_DIR, 'cnn_label_encoder.pkl'))
    print(f"\nModel saved: {ckpt_path}")

    return song_acc, history


if __name__ == '__main__':
    main()
