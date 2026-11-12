
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import kagglehub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def download_gtzan():
    print("Downloading GTZAN dataset from Kaggle...")
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print(f"Dataset downloaded: {path}")
    return path


def find_audio_root(base_path):
    for root, dirs, files in os.walk(base_path):
        wav_files = [f for f in files if f.endswith('.wav') or f.endswith('.au')]
        if wav_files:
            parent = os.path.dirname(root)
            if any(g in os.path.basename(root) for g in GENRES):
                return parent
    return base_path


def extract_features(file_path, duration=30):
    
    try:
        y, sr = librosa.load(file_path, duration=duration, mono=True)

        features = {}

        # 1. MFCC (40 coefficients — mean values)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        for i, val in enumerate(np.mean(mfcc, axis=1)):
            features[f'mfcc_{i+1}'] = val

        # 2. Spectral Centroid (mean)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = np.mean(spectral_centroid)

        # 3. Spectral Rolloff (mean)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)

        # 4. Zero Crossing Rate (mean)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate'] = np.mean(zcr)

        # 5. Chroma Features (12 pitch classes — mean)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i, val in enumerate(np.mean(chroma, axis=1)):
            features[f'chroma_{i+1}'] = val

        # 6. Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)

        return features

    except Exception as e:
        print(f"  ERROR: {file_path} -> {e}")
        return None


def build_features_csv(audio_root):
    rows = []
    total = 0
    errors = 0

    for genre in GENRES:
        genre_path = os.path.join(audio_root, genre)
        if not os.path.exists(genre_path):
            print(f"  WARNING: {genre_path} not found, skipping.")
            continue

        audio_files = [f for f in os.listdir(genre_path)
                       if f.endswith('.wav') or f.endswith('.au')]
        print(f"\n[{genre.upper()}] Processing {len(audio_files)} files...")

        for i, fname in enumerate(sorted(audio_files)):
            fpath = os.path.join(genre_path, fname)
            feats = extract_features(fpath)
            if feats is not None:
                feats['genre'] = genre
                feats['filename'] = fname
                rows.append(feats)
                total += 1
            else:
                errors += 1

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(audio_files)} completed")

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"Total: {total} files processed, {errors} errors")
    print(f"Feature CSV saved: {FEATURES_CSV}")
    print(f"Data shape: {df.shape}")
    return df


def main():
    if os.path.exists(FEATURES_CSV):
        print(f"features.csv already exists: {FEATURES_CSV}")
        print("Delete features.csv to re-extract.")
        return pd.read_csv(FEATURES_CSV)

    dataset_path = download_gtzan()
    audio_root = find_audio_root(dataset_path)
    print(f"Audio root directory: {audio_root}")

    print("\nStarting feature extraction...")
    df = build_features_csv(audio_root)
    return df


if __name__ == '__main__':
    main()
