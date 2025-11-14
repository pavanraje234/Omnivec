#!/usr/bin/env python3
"""
OmniVec Part 4 (UPDATED): OmniVec Architecture (runnable)
- Adds sentiment head
- Fixes emotion label handling (expects integer labels from Part2)
- Ensures OOV -> UNK (1) and PAD -> 0
- Includes robust multi-task loss (sentiment + emotion + reg)
- Dataset now ingests imdb_data for sentiment samples
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import from previous parts
from omnivec_part1 import OmniVecConfig
from omnivec_part2 import TextPreprocessor

# ---------------------------------------------------------------------------
# OmniVec Dataset (fixed)
# ---------------------------------------------------------------------------

class OmniVecDataset(Dataset):
    """Dataset for OmniVec multi-task training (fixed label handling)"""
    
    def __init__(self, corpus_data, emotion_data, causality_data, imdb_data, vocab_to_idx, config):
        self.corpus = corpus_data.get('tokenized', []) if corpus_data else []
        # emotion_data expected to contain integer labels (train/val/test splits)
        self.emotion_texts = emotion_data.get('train_texts', []) if emotion_data else []
        self.emotion_labels = emotion_data.get('train_labels', []) if emotion_data else []
        # IMDB sentiment data (optional)
        self.imdb_texts = imdb_data.get('train_texts', []) if imdb_data else []
        self.imdb_labels = imdb_data.get('train_labels', []) if imdb_data else []
        self.causality_pairs = causality_data.get('cause_effect_pairs', []) if causality_data else []
        
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = {idx: word for word, idx in vocab_to_idx.items()}
        self.config = config
        
        # Configurable params
        self.max_seq = getattr(self.config, 'MAX_SEQ_LEN', 20)
        self.corpus_limit = getattr(self.config, 'CORPUS_LIMIT', None)  # None -> use all
        
        # Prepare samples
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare training samples with multiple task annotations"""
        samples = []
        preprocessor = TextPreprocessor()
        
        # Base corpus samples (unlabeled)
        corpus_iter = self.corpus if self.corpus_limit is None else self.corpus[:self.corpus_limit]
        for tokens in corpus_iter:
            if len(tokens) > 3:
                samples.append({
                    'tokens': tokens,
                    'has_sentiment': False,
                    'sentiment_label': -1,
                    'has_emotion': False,
                    'emotion_label': -1,
                    'has_causality': False,
                    'cause_idx': -1,
                    'effect_idx': -1
                })
        
        # Sentiment (IMDB) samples
        for text, label in zip(self.imdb_texts[:len(self.imdb_texts)], self.imdb_labels[:len(self.imdb_labels)]):
            tokens = preprocessor.tokenize(text)
            if len(tokens) > 2:
                samples.append({
                    'tokens': tokens,
                    'has_sentiment': True,
                    'sentiment_label': int(label),
                    'has_emotion': False,
                    'emotion_label': -1,
                    'has_causality': False,
                    'cause_idx': -1,
                    'effect_idx': -1
                })
        
        # Emotion samples (labels are integers from Part2)
        for text, label in zip(self.emotion_texts[:len(self.emotion_texts)], self.emotion_labels[:len(self.emotion_labels)]):
            tokens = preprocessor.tokenize(text)
            if len(tokens) > 2 and label is not None and int(label) >= 0:
                samples.append({
                    'tokens': tokens,
                    'has_sentiment': False,
                    'sentiment_label': -1,
                    'has_emotion': True,
                    'emotion_label': int(label),
                    'has_causality': False,
                    'cause_idx': -1,
                    'effect_idx': -1
                })
        
        # Causality samples (synthetic)
        for cause, effect, _ in self.causality_pairs:
            cause_tokens = preprocessor.tokenize(cause)
            effect_tokens = preprocessor.tokenize(effect)
            combined = cause_tokens + ['causes'] + effect_tokens
            samples.append({
                'tokens': combined,
                'has_sentiment': False,
                'sentiment_label': -1,
                'has_emotion': False,
                'emotion_label': -1,
                'has_causality': True,
                'cause_idx': 0,
                'effect_idx': len(cause_tokens) + 1
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample['tokens']
        max_seq = self.max_seq
        
        # Convert tokens to indices, map unknown -> UNK (1), PAD=0
        token_indices = [self.vocab_to_idx.get(t, 1) for t in tokens[:max_seq]]
        
        # Pad sequences to max_seq with PAD idx 0
        if len(token_indices) < max_seq:
            token_indices += [0] * (max_seq - len(token_indices))
        
        return {
            'token_indices': torch.LongTensor(token_indices),
            'length': min(len(tokens), max_seq),
            'has_sentiment': torch.tensor(sample.get('has_sentiment', False), dtype=torch.bool),
            'sentiment_label': torch.tensor(sample.get('sentiment_label', -1), dtype=torch.long),
            'has_emotion': torch.tensor(sample.get('has_emotion', False), dtype=torch.bool),
            'emotion_label': torch.tensor(sample.get('emotion_label', -1), dtype=torch.long),
            'has_causality': torch.tensor(sample.get('has_causality', False), dtype=torch.bool),
            'cause_idx': torch.tensor(sample.get('cause_idx', 0), dtype=torch.long),
            'effect_idx': torch.tensor(sample.get('effect_idx', 0), dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# OmniVec Model (added sentiment head)
# ---------------------------------------------------------------------------

class OmniVecModel(nn.Module):
    """
    OmniVec: Unified embedding framework with multiple components
    - Base word embeddings
    - Emotion-aware layer
    - Temporal encoding
    - Causality modeling
    - Sentiment classifier (NEW)
    """
    
    def __init__(self, vocab_size, config):
        super(OmniVecModel, self).__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        
        # === Core Embedding Layer ===
        self.word_embeddings = nn.Embedding(
            vocab_size, 
            config.EMBEDDING_DIM,
            padding_idx=0
        )
        
        # === Emotion Component ===
        self.emotion_encoder = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, config.EMOTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.emotion_classifier = nn.Linear(config.EMOTION_DIM, config.NUM_EMOTIONS)
        self.emotion_refine = nn.Linear(config.EMOTION_DIM, config.EMBEDDING_DIM)
        
        # === Sentiment Component (NEW) ===
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.sentiment_classifier = nn.Linear(config.EMBEDDING_DIM // 2, 2)  # binary sentiment
        
        # === Temporal Component ===
        self.temporal_encoder = nn.Embedding(10, config.TEMPORAL_DIM)
        self.temporal_projection = nn.Linear(
            config.EMBEDDING_DIM + config.TEMPORAL_DIM,
            config.EMBEDDING_DIM
        )
        
        # === Causality Component ===
        self.cause_encoder = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        self.effect_encoder = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        self.causality_scorer = nn.Bilinear(config.EMBEDDING_DIM, config.EMBEDDING_DIM, 1)
        
        # === Context Encoder (for sentence representation) ===
        self.context_lstm = nn.LSTM(
            config.EMBEDDING_DIM,
            config.EMBEDDING_DIM // 2,
            batch_first=True,
            bidirectional=True
        )
        
        # === Skip-gram Prediction Head ===
        self.skipgram_output = nn.Linear(config.EMBEDDING_DIM, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.uniform_(self.word_embeddings.weight, -0.1, 0.1)
        nn.init.constant_(self.word_embeddings.weight[0], 0)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                try:
                    nn.init.xavier_uniform_(module.weight)
                except Exception:
                    pass
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, token_indices, time_idx=0):
        """
        Forward pass
        Args:
            token_indices: [batch, seq_len] token indices
            time_idx: temporal index (0-9)
        Returns:
            Dictionary with embeddings and predictions
        """
        batch_size, seq_len = token_indices.shape
        
        # === Base Embeddings ===
        word_embeds = self.word_embeddings(token_indices)  # [B, L, D]
        
        # === Context Encoding ===
        lstm_out, (hidden, _) = self.context_lstm(word_embeds)
        # hidden: [num_layers*2, B, hidden_dim]; take last layer's forward/backward
        # For 1-layer bidir LSTM hidden[0] and hidden[1] are forward/backward
        if hidden.shape[0] >= 2:
            context_vec = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, D]
        else:
            context_vec = torch.cat([hidden[0], hidden[1]], dim=1)
        
        # === Emotion Processing ===
        emotion_features = self.emotion_encoder(context_vec)
        emotion_logits = self.emotion_classifier(emotion_features)
        
        # Refine embeddings with emotion
        emotion_refined = self.emotion_refine(emotion_features)
        enhanced_context = context_vec + 0.3 * emotion_refined
        
        # === Sentiment Processing (NEW) ===
        sentiment_features = self.sentiment_encoder(context_vec)
        sentiment_logits = self.sentiment_classifier(sentiment_features)
        
        # === Temporal Encoding ===
        time_tensor = torch.full((batch_size,), time_idx, dtype=torch.long, device=token_indices.device)
        temporal_embeds = self.temporal_encoder(time_tensor)
        
        # Combine with temporal info
        combined = torch.cat([enhanced_context, temporal_embeds], dim=1)
        temporal_context = self.temporal_projection(combined)
        
        # === Skip-gram Prediction ===
        skipgram_logits = self.skipgram_output(temporal_context)
        
        return {
            'word_embeddings': word_embeds,
            'context_vec': temporal_context,
            'emotion_logits': emotion_logits,
            'sentiment_logits': sentiment_logits,
            'skipgram_logits': skipgram_logits,
            'lstm_out': lstm_out
        }
    
    def compute_causality_score(self, cause_indices, effect_indices):
        """Compute causality score between cause and effect words"""
        cause_embeds = self.word_embeddings(cause_indices)
        effect_embeds = self.word_embeddings(effect_indices)
        
        cause_encoded = self.cause_encoder(cause_embeds)
        effect_encoded = self.effect_encoder(effect_embeds)
        
        score = self.causality_scorer(cause_encoded, effect_encoded)
        return score
    
    def get_word_embedding(self, word_idx):
        """Get final embedding for a word"""
        word_tensor = torch.LongTensor([word_idx]).to(self.word_embeddings.weight.device)
        return self.word_embeddings(word_tensor).detach().cpu().numpy()[0]
    
    def get_vocabulary_embeddings(self):
        """Get all word embeddings as numpy array"""
        return self.word_embeddings.weight.detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Multi-Task Loss (includes sentiment + emotion + regularization)
# ---------------------------------------------------------------------------

class OmniVecLoss(nn.Module):
    """Multi-task loss combining all OmniVec objectives (includes sentiment)"""
    
    def __init__(self, config):
        super(OmniVecLoss, self).__init__()
        self.config = config
        # loss functions
        self.sentiment_loss = nn.CrossEntropyLoss()
        self.emotion_loss = nn.CrossEntropyLoss()
        self.causality_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets, batch_data):
        """
        Compute combined loss and return losses dict
        outputs: model outputs
        batch_data: batch dict with masks/labels (tensors)
        """
        device = outputs['context_vec'].device
        losses = {}
        total = torch.tensor(0.0, device=device)
        
        # safe retrieval of alphas with defaults
        alpha_sent = getattr(self.config, 'ALPHA_SENTIMENT', 0.6)
        alpha_emo = getattr(self.config, 'ALPHA_EMOTION', 0.3)
        alpha_causal = getattr(self.config, 'ALPHA_CAUSAL', 0.05)
        
        # Sentiment loss
        has_sent = batch_data.get('has_sentiment', None)
        if has_sent is not None and has_sent.any():
            mask = has_sent.bool()
            logits = outputs['sentiment_logits'][mask]
            labels = batch_data['sentiment_label'][mask].to(device)
            if labels.numel() > 0:
                s_loss = self.sentiment_loss(logits, labels)
                losses['sentiment'] = s_loss
                total = total + (alpha_sent * s_loss)
        
        # Emotion loss
        has_emotion = batch_data.get('has_emotion', None)
        if has_emotion is not None and has_emotion.any():
            mask = has_emotion.bool()
            logits = outputs['emotion_logits'][mask]
            labels = batch_data['emotion_label'][mask].to(device)
            if labels.numel() > 0:
                e_loss = self.emotion_loss(logits, labels)
                losses['emotion'] = e_loss
                total = total + (alpha_emo * e_loss)
        
        # Causality loss (placeholder - expects proper labels in batch_data)
        has_caus = batch_data.get('has_causality', None)
        if has_caus is not None and has_caus.any():
            # If Part5 provides causality labels, compute BCEWithLogitsLoss here
            try:
                mask = has_caus.bool()
                if mask.sum() > 0 and 'causality_label' in batch_data:
                    # Example: compute scores from bilinear scorer on first token positions
                    cause_idx = batch_data['cause_idx'][mask].to(device)
                    effect_idx = batch_data['effect_idx'][mask].to(device)
                    # map indices to embeddings and compute score
                    scores = self._compute_causality_scores_from_batch(outputs, cause_idx, effect_idx)
                    caus_labels = batch_data['causality_label'][mask].float().to(device)
                    c_loss = self.causality_loss(scores.squeeze(-1), caus_labels)
                    losses['causal'] = c_loss
                    total = total + (alpha_causal * c_loss)
            except Exception:
                pass
        
        # Regularization (simple L2 on context)
        reg_loss = torch.mean(outputs['context_vec'] ** 2)
        losses['regularization'] = 0.01 * reg_loss
        total = total + (0.01 * reg_loss)
        
        # ensure total is a scalar tensor (requires_grad True)
        losses['total'] = total
        return losses
    
    def _compute_causality_scores_from_batch(self, outputs, cause_idx, effect_idx):
        """
        Helper: compute causality scores for a batch given cause_idx and effect_idx (both tensors)
        This uses the model's embeddings stored in outputs if available. If not available, returns zeros.
        """
        # outputs doesn't directly include word indices; Part5 should compute these externally if needed.
        # Return zeros as placeholder to avoid crashing if not implemented
        device = outputs['context_vec'].device
        return torch.zeros((cause_idx.size(0), 1), device=device)

# ---------------------------------------------------------------------------
# Vocabulary builder (same as before)
# ---------------------------------------------------------------------------

def build_vocabulary(corpus_data, min_count=5):
    """Build vocabulary from corpus"""
    vocab_counter = corpus_data.get('vocab', {})
    
    # Filter by min count
    # Reserve 0: PAD, 1: UNK
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    for word, count in vocab_counter.items():
        if count >= min_count:
            vocab[word] = len(vocab)
    
    print(f"✓ Vocabulary built: {len(vocab)} words (min_count={min_count})")
    return vocab

# ---------------------------------------------------------------------------
# Main: Build dataset, model, and save initial checkpoint
# ---------------------------------------------------------------------------

def main():
    """Main architecture building function"""
    print("\n" + "="*70)
    print("OMNIVEC PROJECT - PART 4: OMNIVEC ARCHITECTURE (UPDATED)")
    print("="*70)
    
    # Initialize config
    config = OmniVecConfig()
    # ensure some useful debug attributes exist
    if not hasattr(config, 'MAX_SEQ_LEN'):
        config.MAX_SEQ_LEN = 20
    if not hasattr(config, 'CORPUS_LIMIT'):
        config.CORPUS_LIMIT = 10000  # default dev limit; set to None for full corpus
    config.set_seed()
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    with open(os.path.join(config.DATA_DIR, 'unified_corpus.pkl'), 'rb') as f:
        unified_corpus = pickle.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'emotion_data.pkl'), 'rb') as f:
        emotion_data = pickle.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'causality_data.pkl'), 'rb') as f:
        causality_data = pickle.load(f)
    
    # Load IMDB (needed for sentiment samples)
    with open(os.path.join(config.DATA_DIR, 'imdb_data.pkl'), 'rb') as f:
        imdb_data = pickle.load(f)
    
    print("✓ Data loaded successfully")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab_to_idx = build_vocabulary(unified_corpus, min_count=config.WORD2VEC_MIN_COUNT)
    vocab_size = len(vocab_to_idx)
    
    # Save vocabulary
    with open(os.path.join(config.MODELS_DIR, 'omnivec_vocab.pkl'), 'wb') as f:
        pickle.dump(vocab_to_idx, f)
    print("✓ Vocabulary saved")
    
    # Create dataset
    print("\nCreating OmniVec dataset...")
    omnivec_dataset = OmniVecDataset(
        unified_corpus,
        emotion_data,
        causality_data,
        imdb_data,
        vocab_to_idx,
        config
    )
    
    print(f"✓ Dataset created: {len(omnivec_dataset)} samples")
    
    # Create DataLoader
    train_loader = DataLoader(
        omnivec_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ DataLoader ready: {len(train_loader)} batches")
    
    # Initialize model
    print("\nInitializing OmniVec model...")
    omnivec_model = OmniVecModel(vocab_size, config).to(config.DEVICE)
    
    # Initialize loss
    criterion = OmniVecLoss(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in omnivec_model.parameters())
    trainable_params = sum(p.numel() for p in omnivec_model.parameters() if p.requires_grad)
    
    print("\n" + "="*70)
    print("OMNIVEC MODEL INITIALIZED")
    print("="*70)
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {config.DEVICE}")
    print("\nModel Components:")
    print(f"  ✓ Word embeddings")
    print(f"  ✓ Emotion encoder & classifier ({config.NUM_EMOTIONS} emotions)")
    print(f"  ✓ Sentiment encoder & classifier (binary)")
    print(f"  ✓ Temporal encoder ({config.TEMPORAL_DIM}-dim)")
    print(f"  ✓ Causality scorer (placeholder)")
    print(f"  ✓ Context LSTM")
    print(f"  ✓ Skip-gram prediction head")
    
    # Save initial model
    torch.save({
        'model_state_dict': omnivec_model.state_dict(),
        'vocab_size': vocab_size,
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    }, os.path.join(config.MODELS_DIR, 'omnivec_initial.pth'))
    
    print("\n✓ Initial model saved -> models/omnivec_initial.pth")
    
    # Save dataset info
    dataset_info = {
        'num_samples': len(omnivec_dataset),
        'num_batches': len(train_loader),
        'vocab_size': vocab_size,
        'batch_size': config.BATCH_SIZE
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'dataset_info.pkl'), 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print("\n" + "="*70)
    print("PART 4 COMPLETE - OMNIVEC ARCHITECTURE (UPDATED)")
    print("="*70)
    print("\nArchitecture highlights: (updated)")
    print("  1. Multi-task learning with sentiment & emotion explicitly wired")
    print("  2. LSTM-based context encoding for sentence representations")
    print("  3. Emotion-aware embedding refinement")
    print("  4. Temporal encoding for diachronic semantics")
    print("  5. Causality modeling scaffold (implement labels in Part5)")
    
    print("\nNext step: Run python omnivec_part5.py (Training & Evaluation) - send it here and I'll patch it to use the new loss & logging.")
    
if __name__ == "__main__":
    main()
