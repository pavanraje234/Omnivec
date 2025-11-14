#!/usr/bin/env python3
"""
OmniVec Part 3: Baseline Models (Word2Vec, FastText)
Train baseline embedding models for comparison
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import Word2Vec, FastText
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy.spatial.distance import cosine

# Import from previous parts
from omnivec_part1 import OmniVecConfig, save_results
from omnivec_part2 import TextPreprocessor

def get_sentence_embedding(model, sentence, method='mean', embedding_dim=200):
    """Get sentence embedding from word embeddings"""
    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize(preprocessor.clean_text(sentence))
    vectors = []
    
    for token in tokens:
        try:
            if hasattr(model, 'wv'):
                vectors.append(model.wv[token])
            else:
                vectors.append(model[token])
        except KeyError:
            continue
    
    if not vectors:
        return np.zeros(embedding_dim)
    
    if method == 'mean':
        return np.mean(vectors, axis=0)
    elif method == 'max':
        return np.max(vectors, axis=0)
    else:
        return np.mean(vectors, axis=0)

from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import numpy as np
import random

def sample_balanced_indices(labels, n_per_class, random_state=42):
    """
    Return a list of indices sampled (without replacement) balanced across classes.
    labels: list-like of integer class labels
    n_per_class: desired number per class (if class has fewer, uses all)
    """
    rng = np.random.RandomState(random_state)
    labels = np.array(labels)
    classes = np.unique(labels)
    indices = []
    for c in classes:
        class_idx = np.where(labels == c)[0].tolist()
        if len(class_idx) == 0:
            continue
        if len(class_idx) <= n_per_class:
            chosen = class_idx
        else:
            chosen = rng.choice(class_idx, size=n_per_class, replace=False).tolist()
        indices.extend(chosen)
    rng.shuffle(indices)
    return indices

def evaluate_sentiment_classification(model, imdb_data, model_name="Model", config=None,
                                      n_per_class_train=2500, n_per_class_test=1000):
    """Evaluate model on IMDB sentiment classification (robust sampling by label)"""
    print(f"\nEvaluating {model_name} on IMDB sentiment...")

    embedding_dim = config.EMBEDDING_DIM if config else 200
    preprocessor = TextPreprocessor()

    # Ensure expected splits exist; fall back if not
    train_texts = imdb_data.get('train_texts', [])
    train_labels = imdb_data.get('train_labels', [])
    val_texts = imdb_data.get('val_texts', [])
    val_labels = imdb_data.get('val_labels', [])
    test_texts = imdb_data.get('test_texts', [])
    test_labels = imdb_data.get('test_labels', [])

    if not train_labels or not test_labels:
        raise ValueError("IMDB data must contain 'train_labels' and 'test_labels'")

    # Sample balanced indices from train and test
    train_idx = sample_balanced_indices(train_labels, n_per_class_train, random_state=config.SEED if config else 42)
    test_idx = sample_balanced_indices(test_labels, n_per_class_test, random_state=(config.SEED+1) if config else 43)

    # Build embeddings
    X_train = []
    y_train = []
    for i in train_idx:
        sent = train_texts[i]
        vec = get_sentence_embedding(model, sent, embedding_dim=embedding_dim)
        X_train.append(vec)
        y_train.append(train_labels[i])
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for i in test_idx:
        sent = test_texts[i]
        vec = get_sentence_embedding(model, sent, embedding_dim=embedding_dim)
        X_test.append(vec)
        y_test.append(test_labels[i])
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    # If any rows are all zeros (no tokens found), print counts
    zero_train = np.sum(np.all(X_train == 0, axis=1))
    zero_test = np.sum(np.all(X_test == 0, axis=1))
    if zero_train or zero_test:
        print(f"Warning: all-zero vectors found - train_zero={zero_train}, test_zero={zero_test}")

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") and clf.classes_.shape[0] == 2 else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = None
    else:
        auc = None
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Detailed report
    clf_report = classification_report(y_test, y_pred, digits=4)
    conf_mat = confusion_matrix(y_test, y_pred)

    print("Classification Report:\n", clf_report)
    print("Confusion Matrix:\n", conf_mat)

    # Save reports
    results_folder = config.RESULTS_DIR if config else './results'
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, f'{model_name}_imdb_report.json'), 'w') as fo:
        json.dump({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': conf_mat.tolist()
        }, fo, indent=2)

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc) if auc is not None else None
    }

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC:   {auc:.4f}")

    return results

def evaluate_emotion_classification(model, emotion_data, model_name="Model", config=None,
                                    n_train_samples=3000, n_test_samples=1000):
    """Evaluate model on emotion classification (robust sampling)"""
    print(f"\nEvaluating {model_name} on emotion classification...")

    embedding_dim = config.EMBEDDING_DIM if config else 200

    train_texts = emotion_data.get('train_texts', [])
    train_labels = emotion_data.get('train_labels', [])
    val_texts = emotion_data.get('val_texts', [])
    val_labels = emotion_data.get('val_labels', [])
    test_texts = emotion_data.get('test_texts', [])
    test_labels = emotion_data.get('test_labels', [])

    if not train_labels or not test_labels:
        raise ValueError("Emotion data must contain 'train_labels' and 'test_labels'")

    # Balanced sample from train and test
    train_idx = sample_balanced_indices(train_labels, int(n_train_samples / max(1, len(set(train_labels)))), random_state=config.SEED if config else 42)
    test_idx = sample_balanced_indices(test_labels, int(n_test_samples / max(1, len(set(test_labels)))), random_state=(config.SEED+2) if config else 44)

    X_train = np.vstack([get_sentence_embedding(model, train_texts[i], embedding_dim=embedding_dim) for i in train_idx])
    y_train = np.array([train_labels[i] for i in train_idx])

    X_test = np.vstack([get_sentence_embedding(model, test_texts[i], embedding_dim=embedding_dim) for i in test_idx])
    y_test = np.array([test_labels[i] for i in test_idx])

    # Check zero vectors
    zero_train = np.sum(np.all(X_train == 0, axis=1))
    zero_test = np.sum(np.all(X_test == 0, axis=1))
    if zero_train or zero_test:
        print(f"Warning: all-zero vectors found - emo_train_zero={zero_train}, emo_test_zero={zero_test}")

    # Train classifier (multiclass)
    clf = LogisticRegression(max_iter=2000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # Detailed report
    clf_report = classification_report(y_test, y_pred, digits=4)
    conf_mat = confusion_matrix(y_test, y_pred)

    print("Classification Report:\n", clf_report)
    print("Confusion Matrix:\n", conf_mat)

    # Save
    results_folder = config.RESULTS_DIR if config else './results'
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, f'{model_name}_emotion_report.json'), 'w') as fo:
        json.dump({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': conf_mat.tolist()
        }, fo, indent=2)

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return results

def compute_word_similarity(model, word1, word2):
    """Compute cosine similarity between two words"""
    try:
        if hasattr(model, 'wv'):
            return model.wv.similarity(word1, word2)
        else:
            v1 = model[word1]
            v2 = model[word2]
            return 1 - cosine(v1, v2)
    except:
        return 0.0

def train_word2vec_cbow(tokenized_corpus, config):
    """Train Word2Vec with CBOW architecture"""
    print("\n" + "="*70)
    print("Training Word2Vec (CBOW) Baseline")
    print("="*70)
    
    start_time = time.time()
    
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=config.EMBEDDING_DIM,
        window=config.WORD2VEC_WINDOW,
        min_count=config.WORD2VEC_MIN_COUNT,
        workers=4,
        sg=0,  # CBOW
        epochs=config.WORD2VEC_EPOCHS,
        negative=config.WORD2VEC_NEGATIVE,
        seed=config.SEED
    )
    
    training_time = time.time() - start_time
    
    # Save model
    model.save(os.path.join(config.MODELS_DIR, 'word2vec_cbow.model'))
    
    print(f"✓ Word2Vec CBOW trained in {training_time:.2f}s")
    print(f"  - Vocabulary size: {len(model.wv)}")
    print(f"  - Vector dimension: {model.vector_size}")
    
    return model, training_time

def train_word2vec_skipgram(tokenized_corpus, config):
    """Train Word2Vec with Skip-gram architecture"""
    print("\n" + "="*70)
    print("Training Word2Vec (Skip-gram) Baseline")
    print("="*70)
    
    start_time = time.time()
    
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=config.EMBEDDING_DIM,
        window=config.WORD2VEC_WINDOW,
        min_count=config.WORD2VEC_MIN_COUNT,
        workers=4,
        sg=1,  # Skip-gram
        epochs=config.WORD2VEC_EPOCHS,
        negative=config.WORD2VEC_NEGATIVE,
        seed=config.SEED
    )
    
    training_time = time.time() - start_time
    
    # Save model
    model.save(os.path.join(config.MODELS_DIR, 'word2vec_skipgram.model'))
    
    print(f"✓ Word2Vec Skip-gram trained in {training_time:.2f}s")
    print(f"  - Vocabulary size: {len(model.wv)}")
    print(f"  - Vector dimension: {model.vector_size}")
    
    return model, training_time

def train_fasttext(tokenized_corpus, config):
    """Train FastText for subword embeddings"""
    print("\n" + "="*70)
    print("Training FastText Baseline")
    print("="*70)
    
    start_time = time.time()
    
    model = FastText(
        sentences=tokenized_corpus,
        vector_size=config.EMBEDDING_DIM,
        window=config.WORD2VEC_WINDOW,
        min_count=config.WORD2VEC_MIN_COUNT,
        workers=4,
        sg=1,  # Skip-gram
        epochs=config.WORD2VEC_EPOCHS,
        seed=config.SEED,
        min_n=3,  # Min character n-gram
        max_n=6,  # Max character n-gram
    )
    
    training_time = time.time() - start_time
    
    # Save model
    model.save(os.path.join(config.MODELS_DIR, 'fasttext.model'))
    
    print(f"✓ FastText trained in {training_time:.2f}s")
    print(f"  - Vocabulary size: {len(model.wv)}")
    print(f"  - Vector dimension: {model.vector_size}")
    print(f"  - Character n-grams: {model.wv.min_n}-{model.wv.max_n}")
    
    return model, training_time

def visualize_baseline_comparison(baseline_results, config):
    """Create visualization comparing baseline results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['W2V\nCBOW', 'W2V\nSG', 'FastText']
    
    # Sentiment metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    sentiment_scores = [
        [baseline_results['word2vec_cbow']['sentiment'][m] for m in metrics],
        [baseline_results['word2vec_skipgram']['sentiment'][m] for m in metrics],
        [baseline_results['fasttext']['sentiment'][m] for m in metrics]
    ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (model, scores) in enumerate(zip(models, sentiment_scores)):
        axes[0].bar(x + i*width, scores, width, label=model)
    
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Sentiment Classification Performance')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Emotion metrics
    emotion_metrics = ['accuracy', 'precision', 'recall', 'f1']
    emotion_scores = [
        [baseline_results['word2vec_cbow']['emotion'][m] for m in emotion_metrics],
        [baseline_results['word2vec_skipgram']['emotion'][m] for m in emotion_metrics],
        [baseline_results['fasttext']['emotion'][m] for m in emotion_metrics]
    ]
    
    x = np.arange(len(emotion_metrics))
    
    for i, (model, scores) in enumerate(zip(models, emotion_scores)):
        axes[1].bar(x + i*width, scores, width, label=model)
    
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Emotion Classification Performance')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(emotion_metrics)
    axes[1].legend()
    axes[1].set_ylim([0.3, 0.8])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {config.PLOTS_DIR}/baseline_comparison.png")

def main():
    """Main baseline training function"""
    print("\n" + "="*70)
    print("OMNIVEC PROJECT - PART 3: BASELINE MODELS")
    print("="*70)
    
    # Initialize config
    config = OmniVecConfig()
    config.set_seed()
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    with open(os.path.join(config.DATA_DIR, 'unified_corpus.pkl'), 'rb') as f:
        unified_corpus = pickle.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'imdb_data.pkl'), 'rb') as f:
        imdb_data = pickle.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'emotion_data.pkl'), 'rb') as f:
        emotion_data = pickle.load(f)
    
    tokenized_corpus = unified_corpus['tokenized']
    print(f"✓ Loaded {len(tokenized_corpus)} tokenized sentences")
    
    # Train baseline models
    w2v_cbow_model, w2v_cbow_time = train_word2vec_cbow(tokenized_corpus, config)
    w2v_sg_model, w2v_sg_time = train_word2vec_skipgram(tokenized_corpus, config)
    fasttext_model, fasttext_time = train_fasttext(tokenized_corpus, config)
    
    # Evaluate all models
    baseline_results = {}
    
    # Word2Vec CBOW
    print("\n" + "="*70)
    print("EVALUATING WORD2VEC CBOW")
    print("="*70)
    w2v_cbow_sentiment = evaluate_sentiment_classification(w2v_cbow_model, imdb_data, "Word2Vec CBOW", config)
    w2v_cbow_emotion = evaluate_emotion_classification(w2v_cbow_model, emotion_data, "Word2Vec CBOW", config)
    
    baseline_results['word2vec_cbow'] = {
        'sentiment': w2v_cbow_sentiment,
        'emotion': w2v_cbow_emotion,
        'training_time': w2v_cbow_time
    }
    
    # Word2Vec Skip-gram
    print("\n" + "="*70)
    print("EVALUATING WORD2VEC SKIP-GRAM")
    print("="*70)
    w2v_sg_sentiment = evaluate_sentiment_classification(w2v_sg_model, imdb_data, "Word2Vec Skip-gram", config)
    w2v_sg_emotion = evaluate_emotion_classification(w2v_sg_model, emotion_data, "Word2Vec Skip-gram", config)
    
    baseline_results['word2vec_skipgram'] = {
        'sentiment': w2v_sg_sentiment,
        'emotion': w2v_sg_emotion,
        'training_time': w2v_sg_time
    }
    
    # FastText
    print("\n" + "="*70)
    print("EVALUATING FASTTEXT")
    print("="*70)
    fasttext_sentiment = evaluate_sentiment_classification(fasttext_model, imdb_data, "FastText", config)
    fasttext_emotion = evaluate_emotion_classification(fasttext_model, emotion_data, "FastText", config)
    
    baseline_results['fasttext'] = {
        'sentiment': fasttext_sentiment,
        'emotion': fasttext_emotion,
        'training_time': fasttext_time
    }
    
    # Word similarity analysis
    print("\n" + "="*70)
    print("WORD SIMILARITY ANALYSIS")
    print("="*70)
    
    test_pairs = [
        ('happy', 'joy'),
        ('sad', 'depressed'),
        ('good', 'excellent'),
        ('bad', 'terrible'),
        ('love', 'hate'),
        ('king', 'queen'),
        ('doctor', 'nurse'),
    ]
    
    print("\nWord Similarities (Cosine):")
    print(f"{'Word Pair':<25} {'W2V-CBOW':<12} {'W2V-SG':<12} {'FastText':<12}")
    print("-" * 65)
    
    for w1, w2 in test_pairs:
        sim_cbow = compute_word_similarity(w2v_cbow_model, w1, w2)
        sim_sg = compute_word_similarity(w2v_sg_model, w1, w2)
        sim_ft = compute_word_similarity(fasttext_model, w1, w2)
        print(f"{w1}-{w2:<20} {sim_cbow:>10.4f}  {sim_sg:>10.4f}  {sim_ft:>10.4f}")
    
    # Save results
    save_results(baseline_results, 'baseline_results.json', config.RESULTS_DIR)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': ['Word2Vec CBOW', 'Word2Vec Skip-gram', 'FastText'],
        'Sentiment Acc': [
            baseline_results['word2vec_cbow']['sentiment']['accuracy'],
            baseline_results['word2vec_skipgram']['sentiment']['accuracy'],
            baseline_results['fasttext']['sentiment']['accuracy']
        ],
        'Sentiment F1': [
            baseline_results['word2vec_cbow']['sentiment']['f1'],
            baseline_results['word2vec_skipgram']['sentiment']['f1'],
            baseline_results['fasttext']['sentiment']['f1']
        ],
        'Emotion Acc': [
            baseline_results['word2vec_cbow']['emotion']['accuracy'],
            baseline_results['word2vec_skipgram']['emotion']['accuracy'],
            baseline_results['fasttext']['emotion']['accuracy']
        ],
        'Emotion F1': [
            baseline_results['word2vec_cbow']['emotion']['f1'],
            baseline_results['word2vec_skipgram']['emotion']['f1'],
            baseline_results['fasttext']['emotion']['f1']
        ],
        'Training Time (s)': [
            baseline_results['word2vec_cbow']['training_time'],
            baseline_results['word2vec_skipgram']['training_time'],
            baseline_results['fasttext']['training_time']
        ]
    })
    
    print("\n" + "="*70)
    print("BASELINE RESULTS SUMMARY")
    print("="*70)
    print(comparison_df.to_string(index=False))
    
    # Save table
    comparison_df.to_csv(os.path.join(config.RESULTS_DIR, 'baseline_comparison.csv'), index=False)
    
    # Create visualization
    visualize_baseline_comparison(baseline_results, config)
    
    # Summary
    print("\n" + "="*70)
    print("PART 3 COMPLETE - BASELINE MODELS")
    print("="*70)
    print("\nKey Findings:")
    print(f"  Best Sentiment Model: {comparison_df.loc[comparison_df['Sentiment Acc'].idxmax(), 'Model']}")
    print(f"  Best Emotion Model: {comparison_df.loc[comparison_df['Emotion Acc'].idxmax(), 'Model']}")
    print(f"  Average Sentiment Accuracy: {comparison_df['Sentiment Acc'].mean():.4f}")
    print(f"  Average Emotion Accuracy: {comparison_df['Emotion Acc'].mean():.4f}")
    print("\nThese are the baselines that OmniVec must outperform!")
    
    print("\nNext step: Run python omnivec_part4.py (OmniVec Architecture)")

if __name__ == "__main__":
    main()
