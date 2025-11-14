#!/usr/bin/env python3
"""
OmniVec Part 2: Data Collection and Preprocessing
Loads multiple datasets for sentiment, emotion, temporal, and multimodal tasks
"""

import os
import pickle
import re
import string
from collections import Counter
from tqdm import tqdm
import emoji
import contractions
from datasets import load_dataset
from nltk.corpus import stopwords

# Import config from part 1
from omnivec_part1 import OmniVecConfig

class TextPreprocessor:
    """Advanced text preprocessing for OmniVec"""
    
    def __init__(self):
        self.emoji_dict = self._build_emoji_dict()
        self.slang_dict = self._build_slang_dict()
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
    
    def _build_emoji_dict(self):
        """Map emojis to textual descriptions"""
        return {
            '😊': 'happy', '😃': 'happy', '😄': 'happy', '😁': 'happy',
            '😢': 'sad', '😭': 'crying', '😔': 'sad',
            '😠': 'angry', '😡': 'angry', '🤬': 'angry',
            '😱': 'scared', '😨': 'fear', '😰': 'anxious',
            '❤️': 'love', '💕': 'love', '💖': 'love',
            '😮': 'surprised', '😲': 'surprised',
            '👍': 'good', '👎': 'bad',
        }
    
    def _build_slang_dict(self):
        """Common slang normalization"""
        return {
            'u': 'you', 'ur': 'your', 'r': 'are',
            'y': 'why', 'b4': 'before', 'gr8': 'great',
            'lol': 'laughing', 'omg': 'oh my god',
            'btw': 'by the way', 'idk': 'i do not know',
            'imo': 'in my opinion', 'tbh': 'to be honest',
            'wtf': 'what the fuck', 'fyi': 'for your information',
            'asap': 'as soon as possible', 'brb': 'be right back',
        }
    
    def clean_text(self, text, expand_contractions=True, handle_emoji=True, 
                   normalize_slang=True, lowercase=True, remove_punct=False):
        """Comprehensive text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Expand contractions
        if expand_contractions:
            text = contractions.fix(text)
        
        # Handle emojis
        if handle_emoji:
            for emo, word in self.emoji_dict.items():
                text = text.replace(emo, f' {word} ')
            # Demojize remaining emojis
            text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize slang
        if normalize_slang:
            tokens = text.split()
            tokens = [self.slang_dict.get(t, t) for t in tokens]
            text = ' '.join(tokens)
        
        # Remove punctuation if requested
        if remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def tokenize(self, text):
        """Simple whitespace tokenization"""
        return text.split()

def load_imdb_data(config, val_size=0.1, random_state=None):
    """Load and preprocess IMDB movie reviews with validation split"""
    print("\n" + "="*70)
    print("Loading IMDB Dataset (with validation split)")
    print("="*70)
    
    preprocessor = TextPreprocessor()
    random_state = random_state or config.SEED
    
    try:
        dataset = load_dataset('imdb')
        
        texts = []
        labels = []
        for item in tqdm(dataset['train'], desc="Processing IMDB train"):
            text = preprocessor.clean_text(item['text'])
            if len(text) > 10:
                texts.append(text)
                labels.append(int(item['label']))
        
        # Stratified split into train/val
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_size, random_state=random_state, stratify=labels
        )
        
        # Process official test split
        test_texts = []
        test_labels = []
        for item in tqdm(dataset['test'], desc="Processing IMDB test"):
            text = preprocessor.clean_text(item['text'])
            if len(text) > 10:
                test_texts.append(text)
                test_labels.append(int(item['label']))
        
        print(f"✓ IMDB loaded: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test samples")
        from collections import Counter
        print("Train label counts:", Counter(train_labels))
        print("Val   label counts:", Counter(val_labels))
        print("Test  label counts:", Counter(test_labels))
        
        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'test_texts': test_texts,
            'test_labels': test_labels
        }
    except Exception as e:
        print(f"Error loading IMDB: {e}")
        return None

from sklearn.model_selection import train_test_split

def load_emotion_data(config, val_size=0.1, random_state=None):
    """Load emotion classification dataset and return integer labels + validation split"""
    print("\n" + "="*70)
    print("Loading Emotion Dataset (with integer labels and val split)")
    print("="*70)
    
    preprocessor = TextPreprocessor()
    random_state = random_state or config.SEED
    
    try:
        dataset = load_dataset('emotion')
        # HuggingFace emotion mapping (0..5) -> names
        hf_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        
        # Choose emotion list from config if it matches; otherwise derive from hf_map
        if hasattr(config, 'EMOTIONS') and set(config.EMOTIONS).issuperset(set(hf_map.values())):
            emotions_list = list(hf_map.values())
        else:
            emotions_list = list(hf_map.values())
        
        name_to_idx = {name: idx for idx, name in enumerate(emotions_list)}
        
        texts = []
        labels = []
        for item in tqdm(dataset['train'], desc="Processing emotion train"):
            text = preprocessor.clean_text(item['text'])
            if len(text) > 2:
                lbl_name = hf_map[item['label']]
                if lbl_name in name_to_idx:
                    texts.append(text)
                    labels.append(name_to_idx[lbl_name])
        
        # Create stratified validation split from train
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_size, random_state=random_state, stratify=labels
        )
        
        # Process test data similarly (map to indices)
        test_texts = []
        test_labels = []
        for item in tqdm(dataset['test'], desc="Processing emotion test"):
            text = preprocessor.clean_text(item['text'])
            if len(text) > 2:
                lbl_name = hf_map[item['label']]
                if lbl_name in name_to_idx:
                    test_texts.append(text)
                    test_labels.append(name_to_idx[lbl_name])
        
        print(f"✓ Emotion dataset loaded: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test samples")
        
        # Save label map as name->index and index->name
        label_map = {'name_to_idx': name_to_idx, 'idx_to_name': {v:k for k,v in name_to_idx.items()}}
        
        # Print class distribution
        from collections import Counter
        print("Train label counts:", Counter(train_labels))
        print("Val   label counts:", Counter(val_labels))
        print("Test  label counts:", Counter(test_labels))
        
        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels,
            'test_texts': test_texts,
            'test_labels': test_labels,
            'label_map': label_map
        }
    except Exception as e:
        print(f"Error loading emotion dataset: {e}")
        return None

def create_temporal_data():
    """Create synthetic temporal corpus with time-varying semantics"""
    print("\n" + "="*70)
    print("Creating Temporal Corpus")
    print("="*70)
    
    # Simulate semantic shifts across time periods
    time_periods = ['2015', '2017', '2019', '2021', '2023']
    temporal_corpus = {}
    
    # Words that change meaning over time
    evolving_words = {
        'corona': {
            '2015': ['corona beer', 'corona sun', 'solar corona'],
            '2019': ['corona beer', 'corona sun', 'solar corona'],
            '2021': ['coronavirus', 'covid pandemic', 'corona virus'],
            '2023': ['coronavirus', 'post pandemic', 'endemic']
        },
        'tweet': {
            '2015': ['bird tweet', 'twitter post', 'social media'],
            '2019': ['twitter post', 'social media', 'online message'],
            '2021': ['twitter post', 'social media', 'microblog'],
            '2023': ['x post', 'social media', 'microblog']
        },
        'viral': {
            '2015': ['video viral', 'internet meme', 'trending'],
            '2019': ['video viral', 'internet meme', 'trending'],
            '2021': ['virus spread', 'pandemic', 'contagious'],
            '2023': ['video viral', 'trending', 'popular content']
        }
    }
    
    # Generate corpus for each time period
    for period in time_periods:
        sentences = []
        
        # Add base sentences (stable across time)
        base_sentences = [
            "technology advances rapidly every year",
            "people enjoy watching movies and series",
            "climate change affects global weather",
            "education is important for development",
        ] * 50
        
        sentences.extend(base_sentences)
        
        # Add time-specific sentences
        for word, contexts in evolving_words.items():
            if period in contexts:
                for context in contexts[period]:
                    sent = f"people talk about {context} frequently"
                    sentences.append(sent)
                    sentences.append(f"news reports about {context}")
        
        temporal_corpus[period] = sentences
    
    print(f"✓ Temporal corpus created with {len(time_periods)} time periods")
    return temporal_corpus, time_periods

def create_causality_data():
    """Create synthetic cause-effect pairs for training"""
    print("\n" + "="*70)
    print("Creating Causality Dataset")
    print("="*70)
    
    # Explicit cause-effect templates
    cause_effect_pairs = [
        ("rain", "wet ground", "heavy rain causes wet ground"),
        ("exercise", "fitness", "regular exercise improves fitness"),
        ("study", "knowledge", "studying increases knowledge"),
        ("pollution", "climate change", "pollution contributes to climate change"),
        ("virus", "infection", "virus causes infection"),
        ("earthquake", "damage", "earthquake causes structural damage"),
        ("inflation", "prices", "inflation raises prices"),
        ("sleep", "energy", "good sleep provides energy"),
        ("stress", "health problems", "chronic stress leads to health problems"),
        ("practice", "skill", "practice enhances skill"),
        ("advertising", "sales", "advertising increases sales"),
        ("poverty", "crime", "poverty correlates with crime rates"),
        ("education", "employment", "education improves employment chances"),
        ("smoking", "cancer", "smoking increases cancer risk"),
        ("investment", "returns", "investment generates returns"),
    ]
    
    # Generate training sentences
    causal_sentences = []
    cause_words = []
    effect_words = []
    
    templates = [
        "{cause} leads to {effect}",
        "because of {cause} we see {effect}",
        "{cause} results in {effect}",
        "the {cause} caused {effect}",
        "{effect} happens due to {cause}",
        "if {cause} then {effect}",
        "{cause} is the reason for {effect}",
    ]
    
    for cause, effect, sent in cause_effect_pairs:
        causal_sentences.append(sent)
        cause_words.append(cause)
        effect_words.append(effect)
        
        # Generate variations
        for template in templates[:3]:
            causal_sentences.append(template.format(cause=cause, effect=effect))
    
    print(f"✓ Causality dataset created with {len(causal_sentences)} sentences")
    
    return {
        'sentences': causal_sentences,
        'cause_effect_pairs': cause_effect_pairs,
        'cause_words': cause_words,
        'effect_words': effect_words
    }

def build_unified_corpus(imdb_data, emotion_data, temporal_corpus, causality_data):
    """Combine all text data into a unified corpus and preserve metadata"""
    print("\n" + "="*70)
    print("Building Unified Corpus (with metadata)")
    print("="*70)
    
    preprocessor = TextPreprocessor()
    all_sentences = []
    metadata = []  # list of tuples (source, split, original_label_or_None)
    
    # Add IMDB (train + val)
    if imdb_data:
        for txt, lbl in zip(imdb_data.get('train_texts', []), imdb_data.get('train_labels', [])):
            all_sentences.append(txt)
            metadata.append(('imdb', 'train', int(lbl)))
        for txt, lbl in zip(imdb_data.get('val_texts', []), imdb_data.get('val_labels', [])):
            all_sentences.append(txt)
            metadata.append(('imdb', 'val', int(lbl)))
        print(f"  Added {len(imdb_data.get('train_texts', []))} IMDB train and {len(imdb_data.get('val_texts', []))} val sentences")
    
    # Add emotion (train + val)
    if emotion_data:
        for txt, lbl in zip(emotion_data.get('train_texts', []), emotion_data.get('train_labels', [])):
            all_sentences.append(txt)
            metadata.append(('emotion', 'train', int(lbl)))
        for txt, lbl in zip(emotion_data.get('val_texts', []), emotion_data.get('val_labels', [])):
            all_sentences.append(txt)
            metadata.append(('emotion', 'val', int(lbl)))
        print(f"  Added {len(emotion_data.get('train_texts', []))} emotion train and {len(emotion_data.get('val_texts', []))} val sentences")
    
    # Add temporal (no labels) but keep provenance
    for period, sentences in temporal_corpus.items():
        for sent in sentences:
            all_sentences.append(sent)
            metadata.append(('temporal', period, None))
    print(f"  Added {sum(len(s) for s in temporal_corpus.values())} temporal sentences")
    
    # Add causality (no labels, but set source)
    for sent in causality_data['sentences']:
        all_sentences.append(sent)
        metadata.append(('causal', 'synthetic', None))
    print(f"  Added {len(causality_data['sentences'])} causality sentences")
    
    # Tokenize all sentences using the project's preprocessor
    tokenized_corpus = [preprocessor.tokenize(sent) for sent in all_sentences]
    
    # Build vocabulary
    vocab_counter = Counter()
    for tokens in tokenized_corpus:
        vocab_counter.update(tokens)
    
    vocab_size = len(vocab_counter)
    print(f"\n✓ Unified corpus created:")
    print(f"  - Total sentences: {len(all_sentences)}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Most common words: {vocab_counter.most_common(10)}")
    
    return {
        'sentences': all_sentences,
        'tokenized': tokenized_corpus,
        'vocab': vocab_counter,
        'vocab_size': vocab_size,
        'metadata': metadata
    }

def main():
    """Main data collection function"""
    print("\n" + "="*70)
    print("OMNIVEC PROJECT - PART 2: DATA COLLECTION & PREPROCESSING")
    print("="*70)
    
    # Initialize config
    config = OmniVecConfig()
    
    # Load datasets
    imdb_data = load_imdb_data(config)
    emotion_data = load_emotion_data(config)
    temporal_corpus, time_periods = create_temporal_data()
    causality_data = create_causality_data()
    
    # Build unified corpus
    unified_corpus = build_unified_corpus(imdb_data, emotion_data, temporal_corpus, causality_data)
    
    # Save all data
    print("\n" + "="*70)
    print("Saving Preprocessed Data")
    print("="*70)
    
    with open(os.path.join(config.DATA_DIR, 'imdb_data.pkl'), 'wb') as f:
        pickle.dump(imdb_data, f)
    print("✓ Saved imdb_data.pkl")
    
    with open(os.path.join(config.DATA_DIR, 'emotion_data.pkl'), 'wb') as f:
        pickle.dump(emotion_data, f)
    print("✓ Saved emotion_data.pkl")
    
    with open(os.path.join(config.DATA_DIR, 'temporal_corpus.pkl'), 'wb') as f:
        pickle.dump({'corpus': temporal_corpus, 'periods': time_periods}, f)
    print("✓ Saved temporal_corpus.pkl")
    
    with open(os.path.join(config.DATA_DIR, 'causality_data.pkl'), 'wb') as f:
        pickle.dump(causality_data, f)
    print("✓ Saved causality_data.pkl")
    
    with open(os.path.join(config.DATA_DIR, 'unified_corpus.pkl'), 'wb') as f:
        pickle.dump(unified_corpus, f)
    print("✓ Saved unified_corpus.pkl")
    
    # Summary
    print("\n" + "="*70)
    print("PART 2 COMPLETE - DATA COLLECTION & PREPROCESSING")
    print("="*70)
    print("\nDataset Statistics:")
    if imdb_data:
        print(f"  IMDB: {len(imdb_data['train_texts'])} train, {len(imdb_data['test_texts'])} test")
    if emotion_data:
        print(f"  Emotion: {len(emotion_data['train_texts'])} train, {len(emotion_data['test_texts'])} test")
    print(f"  Temporal: {len(time_periods)} time periods")
    print(f"  Causality: {len(causality_data['sentences'])} sentences")
    print(f"  Unified Corpus: {len(unified_corpus['sentences'])} sentences, {unified_corpus['vocab_size']} vocab")
    
    print("\nNext step: Run python omnivec_part3.py (Baseline Models)")

if __name__ == "__main__":
    main()
