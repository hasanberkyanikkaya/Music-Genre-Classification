"""
SVM, Random Forest and KNN models(Classic ML algorithms)

"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_and_prepare_data():
    features_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(features_path)

    X = df.drop(columns=['genre', 'filename'], errors='ignore')
    y = df['genre']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"Training set: {X_train_sc.shape}, Test set: {X_test_sc.shape}")
    print(f"Classes: {list(le.classes_)}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, le


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, le):
    """Trains the model, evaluates it and returns results."""
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {
        'model':    model,
        'name':     name,
        'test_acc': acc,
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'y_pred':   y_pred,
        'y_test':   y_test
    }


def plot_confusion_matrix(result, le, save_path):
    cm      = confusion_matrix(result['y_test'], result['y_pred'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        [f"{result['name']} - Confusion Matrix (Count)",
         f"{result['name']} - Confusion Matrix (Normalized)"],
        ['d', '.2f']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, cbar=True)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual',    fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    names     = [r['name'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    cv_means  = [r['cv_mean'] for r in results]
    cv_stds   = [r['cv_std']  for r in results]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy',
                   color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, cv_means, width, label='CV Accuracy (5-fold)',
                   color='coral', alpha=0.85,
                   yerr=cv_stds, capsize=5)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Comparison - Music Genre Classification', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, le = load_and_prepare_data()

    models = [
        ("SVM",           SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("KNN",           KNeighborsClassifier(n_neighbors=5, metric='euclidean')),
    ]

    results = []
    for name, model in models:
        result = train_and_evaluate(name, model, X_train, X_test, y_train, y_test, le)
        results.append(result)

        cm_path = os.path.join(RESULTS_DIR, f'cm_{name.replace(" ", "_").lower()}.png')
        plot_confusion_matrix(result, le, cm_path)

        model_path = os.path.join(MODELS_DIR, f'{name.replace(" ", "_").lower()}.pkl')
        joblib.dump(model, model_path)

    comparison_path = os.path.join(RESULTS_DIR, 'ml_model_comparison.png')
    plot_model_comparison(results, comparison_path)

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(le,     os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    summary = pd.DataFrame([{
        'Model':        r['name'],
        'Test Accuracy': f"{r['test_acc']*100:.2f}%",
        'CV Accuracy':   f"{r['cv_mean']*100:.2f}% ± {r['cv_std']*100:.2f}%"
    } for r in results])
    print(summary.to_string(index=False))

    best = max(results, key=lambda r: r['test_acc'])
    print(f"\nBest ML model: {best['name']} ({best['test_acc']*100:.2f}%)")

    return results, scaler, le


if __name__ == '__main__':
    main()
