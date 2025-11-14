#!/usr/bin/env python3
"""
OmniVec Part 1: Setup & Dependencies
Author: Research Team
Purpose: Environment setup and configuration for OmniVec project
"""

import subprocess
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def install_packages():
    """Install all required packages for OmniVec"""
    print("="*70)
    print("Installing Required Packages")
    print("="*70)
    
    packages = [
        'torch',
        'torchvision',
        'transformers',
        'gensim',
        'scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'nltk',
        'datasets',
        'pillow',
        'tqdm',
        'scipy',
        'umap-learn',
        'emoji',
        'contractions',
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ Installed {package}")
        except:
            print(f"✗ Failed to install {package}")
    
    print("\n✓ Package installation complete!")

class OmniVecConfig:
    """Configuration for OmniVec experiments - all hyperparameters in one place"""
    
    # Random seeds for reproducibility
    SEED = 42
    
    # Paths
    DATA_DIR = './data'
    RESULTS_DIR = './results'
    MODELS_DIR = './models'
    PLOTS_DIR = './plots'
    
    # Embedding dimensions
    EMBEDDING_DIM = 200
    TEMPORAL_DIM = 50
    EMOTION_DIM = 50
    
    # Training hyperparameters
    WORD2VEC_WINDOW = 5
    WORD2VEC_MIN_COUNT = 5
    WORD2VEC_EPOCHS = 10
    WORD2VEC_NEGATIVE = 10
    
    OMNIVEC_EPOCHS = 15
    BATCH_SIZE = 128
    LEARNING_RATE = 0.003
    
    # Loss weights
    ALPHA_EMOTION = 0.15      # Emotion loss weight
    ALPHA_TEMPORAL = 0.05     # Temporal loss weight
    ALPHA_MULTIMODAL = 0.25  # Multimodal loss weight
    ALPHA_CAUSAL = 0.15      # Causality loss weight
    ALPHA_BASE = 1.0         # Base embedding loss weight
    
    # Emotion categories
    EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love', 'neutral']
    NUM_EMOTIONS = len(EMOTIONS)
    
    def __init__(self):
        """Create necessary directories"""
        import torch
        
        for dir_path in [self.DATA_DIR, self.RESULTS_DIR, self.MODELS_DIR, self.PLOTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Device
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_seed(self):
        """Set random seeds for reproducibility"""
        import torch
        
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    print("\n" + "="*70)
    print("Downloading NLTK Data")
    print("="*70)
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"✓ Downloaded {dataset}")
        except:
            print(f"✗ Failed to download {dataset}")

def set_plotting_style():
    """Set publication-quality plotting style"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10

def save_results(results_dict, filename, results_dir='./results'):
    """Save results to JSON file"""
    import json
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"✓ Results saved to {filepath}")

def load_results(filename, results_dir='./results'):
    """Load results from JSON file"""
    import json
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'r') as f:
        return json.load(f)

def save_model(model, filename, models_dir='./models'):
    """Save PyTorch model"""
    import torch
    
    filepath = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"✓ Model saved to {filepath}")

def load_model(model, filename, models_dir='./models'):
    """Load PyTorch model"""
    import torch
    
    filepath = os.path.join(models_dir, filename)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("OMNIVEC PROJECT - PART 1: SETUP & DEPENDENCIES")
    print("="*70)
    
    # Ask if user wants to install packages
    response = input("\nInstall required packages? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        install_packages()
    
    # Download NLTK data
    download_nltk_data()
    
    # Initialize configuration
    print("\n" + "="*70)
    print("Initializing Configuration")
    print("="*70)
    
    config = OmniVecConfig()
    config.set_seed()
    
    print(f"✓ Configuration initialized")
    print(f"  - Device: {config.DEVICE}")
    print(f"  - Embedding dim: {config.EMBEDDING_DIM}")
    print(f"  - Seed: {config.SEED}")
    print(f"  - Data directory: {config.DATA_DIR}")
    
    # Set plotting stylee
    set_plotting_style()
    print("✓ Plotting style configured")
    
    # Save configuration
    config_dict = {
        'SEED': config.SEED,
        'EMBEDDING_DIM': config.EMBEDDING_DIM,
        'TEMPORAL_DIM': config.TEMPORAL_DIM,
        'EMOTION_DIM': config.EMOTION_DIM,
        'OMNIVEC_EPOCHS': config.OMNIVEC_EPOCHS,
        'BATCH_SIZE': config.BATCH_SIZE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'ALPHA_EMOTION': config.ALPHA_EMOTION,
        'ALPHA_TEMPORAL': config.ALPHA_TEMPORAL,
        'ALPHA_MULTIMODAL': config.ALPHA_MULTIMODAL,
        'ALPHA_CAUSAL': config.ALPHA_CAUSAL,
        'ALPHA_BASE': config.ALPHA_BASE,
    }
    
    save_results(config_dict, 'config.json', config.RESULTS_DIR)
    
    print("\n" + "="*70)
    print("PART 1 COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python omnivec_part2.py  (Data Collection)")
    print("2. Run: python omnivec_part3.py  (Baseline Models)")
    print("3. Run: python omnivec_part4.py  (OmniVec Architecture)")
    print("4. Run: python omnivec_part5.py  (Training & Evaluation)")
    print("5. Run: python omnivec_part6.py  (Analysis & Visualization)")

if __name__ == "__main__":
    main()
