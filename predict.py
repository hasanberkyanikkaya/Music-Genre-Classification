"""
Music Genre Prediction
======================
Usage:
    python predict.py audio_file.wav
    python predict.py audio_file.wav --model cnn
    python predict.py audio_file.wav --model all   (default)

Download + classify:
    Files downloaded to the Musics folder are automatically
    moved to "Classified ones/<genre>/" after prediction.
"""

import os
import sys
import shutil
import argparse
import warnings
import logging

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.disable(logging.CRITICAL)

import numpy as np
import librosa
import joblib

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
MUSICS_DIR      = os.path.join(BASE_DIR, '..', 'Musics')
CLASSIFIED_DIR  = os.path.join(BASE_DIR, '..', 'Classified ones')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

IMG_SIZE    = 64
N_MELS      = 64
SEG_SECONDS = 3
DURATION    = 30
SR          = 22050


# ---------------------------------------------------------------------------
# Feature Extraction (for ML models)
# ---------------------------------------------------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=DURATION, mono=True)
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))

    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    return np.array(features).reshape(1, -1)


# ---------------------------------------------------------------------------
# Mel-Spectrogram (for CNN)
# ---------------------------------------------------------------------------
def segment_to_melspec(segment, sr):
    import tensorflow as tf
    mel    = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS,
                                            hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_rs = tf.image.resize(mel_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE]).numpy()
    mel_n  = (mel_rs - mel_rs.min()) / (mel_rs.max() - mel_rs.min() + 1e-8)
    return mel_n.astype(np.float32)


def extract_segments(file_path):
    audio, sr = librosa.load(file_path, duration=DURATION, mono=True, sr=SR)
    seg_samples = int(sr * SEG_SECONDS)
    segments = []
    for start in range(0, len(audio) - seg_samples + 1, seg_samples):
        seg  = audio[start : start + seg_samples]
        spec = segment_to_melspec(seg, sr)
        segments.append(spec)
    return np.array(segments, dtype=np.float32)


# ---------------------------------------------------------------------------
# ML Model Prediction
# ---------------------------------------------------------------------------
def predict_ml(file_path, model_name):
    model_files = {'svm': 'svm.pkl', 'rf': 'random_forest.pkl', 'knn': 'knn.pkl'}
    model_path  = os.path.join(MODELS_DIR, model_files[model_name])
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    le_path     = os.path.join(MODELS_DIR, 'label_encoder.pkl')

    # Suppress joblib/wmic stderr noise on Windows (covers load + predict)
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le     = joblib.load(le_path)

        X    = extract_features(file_path)
        X_sc = scaler.transform(X)

        pred_idx = model.predict(X_sc)[0]
        genre    = le.inverse_transform([pred_idx])[0]

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_sc)[0]
            top3  = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
        else:
            top3 = [(genre, 1.0)]
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    return genre, top3


# ---------------------------------------------------------------------------
# CNN Prediction
# ---------------------------------------------------------------------------
def predict_cnn(file_path):
    # Suppress TensorFlow and joblib stderr noise
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    sys.stderr.close()
    sys.stderr = _stderr

    model_path = os.path.join(MODELS_DIR, 'cnn_best.keras')
    le_path    = os.path.join(MODELS_DIR, 'cnn_label_encoder.pkl')

    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    model = tf.keras.models.load_model(model_path)
    le    = joblib.load(le_path)
    sys.stderr.close()
    sys.stderr = _stderr

    segments = extract_segments(file_path)
    if len(segments) == 0:
        return "unknown", []

    preds      = model.predict(segments, verbose=0)
    mean_probs = preds.mean(axis=0)
    pred_idx   = np.argmax(mean_probs)
    genre      = le.inverse_transform([pred_idx])[0]

    top3 = sorted(zip(le.classes_, mean_probs), key=lambda x: x[1], reverse=True)[:3]
    return genre, top3


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------
def print_result(model_label, genre, top3):
    print(f"\n{'─'*40}")
    print(f"  Model     : {model_label}")
    print(f"  Prediction: {genre.upper()}")
    if top3 and len(top3) > 1:
        print(f"  Top-3     :")
        for g, p in top3:
            bar = '█' * int(p * 20)
            print(f"    {g:<12} {p*100:5.1f}%  {bar}")
    print(f"{'─'*40}")


# ---------------------------------------------------------------------------
# Determine Final Genre (CNN-first, fallback to majority vote)
# ---------------------------------------------------------------------------
def decide_final_genre(all_results):
    """
    Returns the genre with the highest confidence score.
    Prioritizes CNN result; falls back to majority vote if CNN is absent.
    """
    for label, genre, top3 in all_results:
        if label == 'CNN':
            return genre, top3[0][1] if top3 else 1.0

    from collections import Counter
    votes  = [genre for _, genre, _ in all_results]
    winner = Counter(votes).most_common(1)[0][0]
    return winner, None


# ---------------------------------------------------------------------------
# Move File to "Classified ones/<genre>/"
# ---------------------------------------------------------------------------
def move_to_classified(file_path, genre):
    dest_dir = os.path.normpath(os.path.join(CLASSIFIED_DIR, genre))
    os.makedirs(dest_dir, exist_ok=True)

    fname = os.path.basename(file_path)
    dest  = os.path.join(dest_dir, fname)

    # Avoid overwriting — append number if file already exists
    if os.path.exists(dest):
        base, ext = os.path.splitext(fname)
        i = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{base}_{i}{ext}")
            i += 1

    shutil.move(file_path, dest)
    print(f"\n  File moved →  Classified ones/{genre}/{os.path.basename(dest)}")
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Music Genre Classifier')
    parser.add_argument('audio', nargs='?',
                        help='Audio file path. If omitted, scans the Musics/ folder.')
    parser.add_argument('--model', default='all',
                        choices=['svm', 'rf', 'knn', 'cnn', 'all'],
                        help='Model to use (default: all)')
    parser.add_argument('--no-move', action='store_true',
                        help='Predict only, do not move the file')
    args = parser.parse_args()

    # If no file given, scan Musics/ folder
    if args.audio is None:
        musics_path = os.path.normpath(MUSICS_DIR)
        if not os.path.exists(musics_path):
            print(f"ERROR: Musics folder not found: {musics_path}")
            sys.exit(1)
        audio_files = [
            os.path.join(musics_path, f)
            for f in os.listdir(musics_path)
            if f.lower().endswith(('.wav', '.mp3', '.au', '.flac', '.ogg', '.m4a'))
        ]
        if not audio_files:
            print(f"No audio files found in Musics/: {musics_path}")
            sys.exit(0)
        print(f"\n{len(audio_files)} file(s) found in Musics/ folder.\n")
    else:
        if not os.path.exists(args.audio):
            print(f"ERROR: File not found: {args.audio}")
            sys.exit(1)
        audio_files = [args.audio]

    ml_map = {'svm': 'SVM', 'rf': 'Random Forest', 'knn': 'KNN'}

    for file_path in audio_files:
        print(f"\n{'═'*45}")
        print(f"  File    : {os.path.basename(file_path)}")
        print(f"  Model(s): {args.model}")
        print(f"{'═'*45}")
        print("Analyzing...", flush=True)

        all_results = []

        if args.model == 'all':
            for key, label in ml_map.items():
                genre, top3 = predict_ml(file_path, key)
                print_result(label, genre, top3)
                all_results.append((label, genre, top3))
            genre, top3 = predict_cnn(file_path)
            print_result('CNN', genre, top3)
            all_results.append(('CNN', genre, top3))

        elif args.model in ml_map:
            genre, top3 = predict_ml(file_path, args.model)
            print_result(ml_map[args.model], genre, top3)
            all_results.append((ml_map[args.model], genre, top3))

        elif args.model == 'cnn':
            genre, top3 = predict_cnn(file_path)
            print_result('CNN', genre, top3)
            all_results.append(('CNN', genre, top3))

        # Final decision
        final_genre, confidence = decide_final_genre(all_results)
        conf_str = f" (confidence: {confidence*100:.1f}%)" if confidence else ""
        print(f"\n  ✔  Result : {final_genre.upper()}{conf_str}")

        # Move file
        if not args.no_move:
            move_to_classified(file_path, final_genre)

    print()


if __name__ == '__main__':
    main()
