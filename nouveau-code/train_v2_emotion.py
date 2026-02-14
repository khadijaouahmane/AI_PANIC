"""
EXPECTED RESULTS:
- Training time: 2-3 hours on CPU
- Validation accuracy: 72-78%
- Test accuracy: 70-76%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import json
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import Wav2Vec2
try:
    from transformers import (
        Wav2Vec2Processor,
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Config
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ transformers not installed!")
    print("Run: pip install transformers datasets")
    exit()


# =========================================================
# DATASET SETUP
# =========================================================

class CremaDownloader:
    def __init__(self, data_dir="./crema_data"):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "AudioWAV"

    def download_dataset(self):
        print("Download AudioWAV.zip from:")
        print("https://github.com/CheyneyComputerScience/CREMA-D")
        zip_path = Path("AudioWAV.zip")
        if not zip_path.exists():
            print("❌ AudioWAV.zip not found")
            return False
        self.extract(zip_path)
        return True

    def extract(self, zip_path):
        self.data_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.data_dir)
        print("✓ Dataset extracted")

    def parse_filename(self, file):
        emotion_map = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fear",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        }
        emotion_code = file.stem.split("_")[2]
        return {
            "file_path": str(file),
            "emotion": emotion_map[emotion_code],
        }

    def create_metadata(self):
        files = list(self.audio_dir.glob("*.wav"))
        data = [self.parse_filename(f) for f in files]
        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / "metadata.csv", index=False)
        print(f"✓ Metadata created: {len(df)} samples")
        return df


# =========================================================
# WAV2VEC2 DATASET (Uses RAW AUDIO, not MFCC)
# =========================================================

class Wav2Vec2Dataset(Dataset):
    """
    Uses RAW audio (not MFCC features)
    Wav2Vec2 learns its own features
    """
    
    def __init__(self, paths, labels, processor, sr=16000, max_duration=2.5):  # ✅ reduced max_duration
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.sr = sr
        self.max_samples = int(sr * max_duration)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Load raw audio
        audio, _ = librosa.load(self.paths[idx], sr=self.sr)
        
        # Pad or truncate
        if len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        else:
            audio = audio[:self.max_samples]
        
        # Process for Wav2Vec2
        inputs = self.processor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }


# =========================================================
# SIMPLE TRAINER FOR WAV2VEC2
# =========================================================

class Wav2Vec2Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=3e-5,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=30,
            eta_min=1e-6
        )
        
        self.history = {'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc='Training')
        for batch in pbar:
            inputs = batch['input_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        return total_loss / len(loader), 100. * correct / total
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(loader), 100. * correct / total, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=30):
        best_val_acc = 0
        patience = 0
        max_patience = 7
        
        print("\n" + "="*70)
        print("TRAINING WAV2VEC2 ON CREMA-D")
        print("="*70)
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 70)
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            gap = train_acc - val_acc
            print(f'Train {train_acc:.2f}% | Val {val_acc:.2f}% | Gap {gap:.1f}% | LR {current_lr:.6f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }, 'wav2vec2_crema_best.pth')
                print('✓ Model saved')
            else:
                patience += 1
                print(f'⚠ No improvement ({patience}/{max_patience})')
                if patience >= max_patience:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                    break
        
        print(f'\n{"="*70}')
        print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
        print(f'{"="*70}')
        return self.history
    
    def evaluate(self, test_loader, label_encoder):
        print("\n" + "="*70)
        print("FINAL TEST EVALUATION")
        print("="*70)
        
        test_loss, test_acc, test_preds, test_labels = self.validate(test_loader)
        
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        print("\nPer-Emotion Performance:")
        print("-" * 70)
        
        for i, emotion in enumerate(label_encoder.classes_):
            mask = np.array(test_labels) == i
            if mask.sum() > 0:
                emotion_acc = 100 * (np.array(test_preds)[mask] == i).sum() / mask.sum()
                total = mask.sum()
                print(f"{emotion:10s}: {emotion_acc:5.2f}% ({total} samples)")
        
        print(f"\n{classification_report(test_labels, test_preds, target_names=label_encoder.classes_, digits=3)}")
        
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f'Wav2Vec2 on CREMA-D - Test Accuracy: {test_acc:.2f}%')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('wav2vec2_confusion_matrix.png', dpi=300)
        print("\n✓ Confusion matrix saved: wav2vec2_confusion_matrix.png")
        
        return test_acc


# =========================================================
# MAIN SCRIPT
# =========================================================

def main():
    print("\n" + "="*70)
    print("WAV2VEC2 FINE-TUNING FOR CREMA-D")
    print("Expected Accuracy: 72-78%")
    print("="*70)
    
    if not TRANSFORMERS_AVAILABLE:
        return
    
    downloader = CremaDownloader()
    if not downloader.audio_dir.exists():
        if not downloader.download_dataset():
            return
    
    metadata_path = downloader.data_dir / "metadata.csv"
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        print(f"✓ Loaded metadata: {len(df)} samples")
    else:
        df = downloader.create_metadata()
    
    le = LabelEncoder()
    labels = le.fit_transform(df["emotion"])
    paths = df["file_path"].values
    num_classes = len(le.classes_)
    
    print(f"\nEmotions: {list(le.classes_)}")
    print(f"Total samples: {len(paths)}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    print("\nLoading Wav2Vec2 pre-trained model...")
    model_name = "facebook/wav2vec2-base"
    
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_classes,
        problem_type="single_label_classification"
    )
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    model.freeze_feature_encoder()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Model loaded")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    train_dataset = Wav2Vec2Dataset(X_train, y_train, processor)
    val_dataset = Wav2Vec2Dataset(X_val, y_val, processor)
    test_dataset = Wav2Vec2Dataset(X_test, y_test, processor)
    
    # ✅ Windows-safe DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=8, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=8, num_workers=0)
    
    print(f"✓ Datasets ready")
    
    # ✅ Force CPU
    device = 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = Wav2Vec2Trainer(model, device=device)
    history = trainer.train(train_loader, val_loader, epochs=30)
    
    test_acc = trainer.evaluate(test_loader, le)
    
    with open('wav2vec2_labels.json', 'w') as f:
        json.dump(le.classes_.tolist(), f)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train', marker='o')
    plt.plot(history['val_acc'], label='Validation', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Wav2Vec2 Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('wav2vec2_training_curves.png', dpi=300)
    print("✓ Training curves saved: wav2vec2_training_curves.png")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nFiles saved:")
    print("  - wav2vec2_crema_best.pth (model)")
    print("  - wav2vec2_labels.json (labels)")
    print("  - wav2vec2_confusion_matrix.png (results)")
    print("  - wav2vec2_training_curves.png (progress)")
    print("="*70)


if __name__ == "__main__":
    main()
