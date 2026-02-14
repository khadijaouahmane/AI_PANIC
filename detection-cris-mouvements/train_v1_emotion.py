"""
Speech Emotion Recognition Training - Optimized for CREMA-D Dataset
"""

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATASET DOWNLOADER
# ============================================================================

class CremaDownloader:
    def __init__(self, data_dir='./crema_data'):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / 'AudioWAV'

    def download_dataset(self):
        print("\nDownload AudioWAV.zip from:")
        print("https://github.com/CheyneyComputerScience/CREMA-D\n")
        zip_path = Path('AudioWAV.zip')
        if not zip_path.exists():
            print("❌ AudioWAV.zip not found")
            return False
        self.extract_dataset(zip_path)
        return True

    def extract_dataset(self, zip_path):
        self.data_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.data_dir)
        print(f"✓ Extracted to {self.audio_dir}")

    def parse_filename(self, filename):
        parts = filename.stem.split('_')
        emotion_map = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        return {
            'file_path': str(filename),
            'emotion': emotion_map[parts[2]]
        }

    def create_metadata(self):
        files = list(self.audio_dir.glob('*.wav'))
        data = [self.parse_filename(f) for f in files]
        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / 'metadata.csv', index=False)
        return df


# ============================================================================
# DATASET
# ============================================================================

class CremaDataset(Dataset):
    def __init__(self, paths, labels, sr=16000, duration=3, augment=False):
        self.paths = paths
        self.labels = labels
        self.sr = sr
        self.max_len = sr * duration
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.paths[idx], sr=self.sr)

        if self.augment:
            if np.random.rand() < 0.5:
                audio += 0.005 * np.random.randn(len(audio))

        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)))
        else:
            audio = audio[:self.max_len]

        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(self.labels[idx])


# ============================================================================
# MODEL
# ============================================================================

class CremaEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(120, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.fc(x)


# ============================================================================
# TRAINER
# ============================================================================

class CremaTrainer:
    def __init__(self, model, label_encoder):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.encoder = label_encoder

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    def train_epoch(self, loader):
        self.model.train()
        loss_sum, correct, total = 0, 0, 0

        for x, y in tqdm(loader, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        return loss_sum / len(loader), 100 * correct / total

    def eval(self, loader):
        self.model.eval()
        loss_sum, correct, total = 0, 0, 0
        preds, labels = [], []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                loss_sum += loss.item()
                p = out.argmax(1)
                correct += (p == y).sum().item()
                total += y.size(0)

                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())

        return loss_sum / len(loader), 100 * correct / total, preds, labels

    def train(self, train_loader, val_loader, epochs=50):
        best = 0
        for e in range(epochs):
            print(f"\nEpoch {e+1}/{epochs}")
            tl, ta = self.train_epoch(train_loader)
            vl, va, _, _ = self.eval(val_loader)

            self.scheduler.step(vl)

            print(f"Train {ta:.2f}% | Val {va:.2f}%")

            if va > best:
                best = va
                torch.save(self.model.state_dict(), "crema_best_model.pth")
                print("✓ Model saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    downloader = CremaDownloader()
    if not downloader.audio_dir.exists():
        if not downloader.download_dataset():
            return

    df = downloader.create_metadata()

    le = LabelEncoder()
    labels = le.fit_transform(df['emotion'])
    paths = df['file_path'].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_ds = CremaDataset(X_train, y_train, augment=True)
    val_ds = CremaDataset(X_val, y_val)
    test_ds = CremaDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=2)

    model = CremaEmotionModel(num_classes=len(le.classes_))
    trainer = CremaTrainer(model, le)

    trainer.train(train_loader, val_loader)

    with open("crema_labels.json", "w") as f:
        json.dump(le.classes_.tolist(), f)

    print("\n✓ Training complete")


if __name__ == "__main__":
    main()
