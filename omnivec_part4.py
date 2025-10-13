#!/usr/bin/env python3
"""
OmniVec Part 4: OmniVec Architecture
Novel multi-task embedding framework with emotion, temporal, multimodal, and causality components
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

# ============================================================================
# OmniVec Dataset
# ============================================================================

class OmniVecDataset(Dataset):
    """Dataset for OmniVec multi-task training"""
    
    def __init__(self, corpus_data, emotion_data, causality_data, vocab_to_idx, config):
        self.corpus = corpus_data['tokenized']
        self.emotion_texts = emotion_data['train_texts'] if emotion_data else []
        self.emotion_labels = emotion_data['train_labels'] if emotion_data else []
        self.causality_pairs = causality_data['cause_effect_pairs'] if causality_data else []
        
        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = {idx: word for word, idx in vocab_to_idx.items()}
        self.config = config
        
        # Build emotion label map
        self.emotion_to_idx = {emo: i for i, emo in enumerate(config.EMOTIONS)}
        
        # Prepare samples
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare training samples with multiple task annotations"""
        samples = []
        
        # Base corpus samples
        for tokens in self.corpus[:10000]:  # Limit for efficiency
            if len(tokens) > 3:
                samples.append({
                    'tokens': tokens,
                    'has_emotion': False,
                    'emotion_label': 0,
                    'has_causality': False,
                    'cause_idx': -1,
                    'effect_idx': -1
                })
        
        # Emotion samples
        preprocessor = TextPreprocessor()
        for text, emotion in zip(self.emotion_texts[:3000], self.emotion_labels[:3000]):
            tokens = preprocessor.tokenize(text)
            if len(tokens) > 2 and emotion in self.emotion_to_idx:
                samples.append({
                    'tokens': tokens,
                    'has_emotion': True,
                    'emotion_label': self.emotion_to_idx[emotion],
                    'has_causality': False,
                    'cause_idx': -1,
                    'effect_idx': -1
                })
        
        # Causality samples
        for cause, effect, _ in self.causality_pairs:
            cause_tokens = preprocessor.tokenize(cause)
            effect_tokens = preprocessor.tokenize(effect)
            combined = cause_tokens + ['causes'] + effect_tokens
            
            samples.append({
                'tokens': combined,
                'has_emotion': False,
                'emotion_label': 0,
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
        
        # Convert tokens to indices
        token_indices = [self.vocab_to_idx.get(t, 0) for t in tokens[:20]]
        
        # FIXED: Pad ALL sequences to length 20 (not just those < 5)
        if len(token_indices) < 20:
            token_indices += [0] * (20 - len(token_indices))
        
        return {
            'token_indices': torch.LongTensor(token_indices),
            'length': min(len(tokens), 20),
            'has_emotion': sample['has_emotion'],
            'emotion_label': sample['emotion_label'],
            'has_causality': sample['has_causality'],
            'cause_idx': sample['cause_idx'] if sample['cause_idx'] >= 0 else 0,
            'effect_idx': sample['effect_idx'] if sample['effect_idx'] >= 0 else 0,
        }

# ============================================================================
# OmniVec Model Architecture
# ============================================================================

class OmniVecModel(nn.Module):
    """
    OmniVec: Unified embedding framework with multiple components
    - Base word embeddings
    - Emotion-aware layer
    - Temporal encoding
    - Causality modeling
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
        
        # Emotion-aware embedding refinement
        self.emotion_refine = nn.Linear(
            config.EMOTION_DIM, 
            config.EMBEDDING_DIM
        )
        
        # === Temporal Component ===
        self.temporal_encoder = nn.Embedding(10, config.TEMPORAL_DIM)
        self.temporal_projection = nn.Linear(
            config.EMBEDDING_DIM + config.TEMPORAL_DIM,
            config.EMBEDDING_DIM
        )
        
        # === Causality Component ===
        self.cause_encoder = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        self.effect_encoder = nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        
        # Causality relation scorer
        self.causality_scorer = nn.Bilinear(
            config.EMBEDDING_DIM,
            config.EMBEDDING_DIM,
            1
        )
        
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
                nn.init.xavier_uniform_(module.weight)
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
        word_embeds = self.word_embeddings(token_indices)
        
        # === Context Encoding ===
        lstm_out, (hidden, _) = self.context_lstm(word_embeds)
        context_vec = torch.cat([hidden[0], hidden[1]], dim=1)
        
        # === Emotion Processing ===
        emotion_features = self.emotion_encoder(context_vec)
        emotion_logits = self.emotion_classifier(emotion_features)
        
        # Refine embeddings with emotion
        emotion_refined = self.emotion_refine(emotion_features)
        enhanced_context = context_vec + 0.3 * emotion_refined
        
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

# ============================================================================
# Multi-Task Loss Function
# ============================================================================

class OmniVecLoss(nn.Module):
    """Multi-task loss combining all OmniVec objectives"""
    
    def __init__(self, config):
        super(OmniVecLoss, self).__init__()
        self.config = config
        
        self.skipgram_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.emotion_loss = nn.CrossEntropyLoss()
        self.causality_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets, batch_data):
        """
        Compute combined loss
        Args:
            outputs: Model outputs dictionary
            targets: Target token indices
            batch_data: Batch metadata
        """
        losses = {}
        
        # === Base Skip-gram Loss ===
        if 'skipgram_logits' in outputs and targets is not None:
            sg_loss = self.skipgram_loss(
                outputs['skipgram_logits'],
                targets
            )
            losses['skipgram'] = self.config.ALPHA_BASE * sg_loss
        
        # === Emotion Loss ===
        has_emotion = batch_data.get('has_emotion', None)
        if has_emotion is not None and has_emotion.any():
            emotion_mask = has_emotion.bool()
            if emotion_mask.sum() > 0:
                emotion_logits = outputs['emotion_logits'][emotion_mask]
                emotion_labels = batch_data['emotion_label'][emotion_mask]
                
                emo_loss = self.emotion_loss(emotion_logits, emotion_labels)
                losses['emotion'] = self.config.ALPHA_EMOTION * emo_loss
        
        # === Temporal Smoothness Loss ===
        if 'context_vec' in outputs:
            temp_loss = torch.mean(outputs['context_vec'] ** 2)
            losses['temporal'] = self.config.ALPHA_TEMPORAL * 0.01 * temp_loss
        
        # === Total Loss ===
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses

def build_vocabulary(corpus_data, min_count=5):
    """Build vocabulary from corpus"""
    vocab_counter = corpus_data['vocab']
    
    # Filter by min count
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in vocab_counter.items():
        if count >= min_count:
            vocab[word] = len(vocab)
    
    print(f"✓ Vocabulary built: {len(vocab)} words (min_count={min_count})")
    return vocab

def main():
    """Main architecture building function"""
    print("\n" + "="*70)
    print("OMNIVEC PROJECT - PART 4: OMNIVEC ARCHITECTURE")
    print("="*70)
    
    # Initialize config
    config = OmniVecConfig()
    config.set_seed()
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    with open(os.path.join(config.DATA_DIR, 'unified_corpus.pkl'), 'rb') as f:
        unified_corpus = pickle.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'emotion_data.pkl'), 'rb') as f:
        emotion_data = pickle.load(f)
    
    with open(os.path.join(config.DATA_DIR, 'causality_data.pkl'), 'rb') as f:
        causality_data = pickle.load(f)
    
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
    print(f"  ✓ Temporal encoder ({config.TEMPORAL_DIM}-dim)")
    print(f"  ✓ Causality scorer")
    print(f"  ✓ Context LSTM")
    print(f"  ✓ Skip-gram prediction head")
    
    # Save initial model
    torch.save({
        'model_state_dict': omnivec_model.state_dict(),
        'vocab_size': vocab_size,
        'config': config.__dict__
    }, os.path.join(config.MODELS_DIR, 'omnivec_initial.pth'))
    
    print("\n✓ Initial model saved")
    
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
    print("PART 4 COMPLETE - OMNIVEC ARCHITECTURE")
    print("="*70)
    print("\nArchitecture highlights:")
    print("  1. Multi-task learning with emotion, temporal, and causality signals")
    print("  2. LSTM-based context encoding for better sentence representations")
    print("  3. Emotion-aware embedding refinement")
    print("  4. Temporal encoding for diachronic semantics")
    print("  5. Causality modeling with dedicated cause/effect encoders")
    
    print("\nNext step: Run python omnivec_part5.py (Training & Evaluation)")

if __name__ == "__main__":
    main()
