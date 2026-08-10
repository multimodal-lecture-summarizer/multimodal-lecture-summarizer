import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.models.binary_classifiers import SlideTextMatcher, KeyframeMatcher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BinaryMatchingDataset(Dataset):
    def __init__(self, num_samples=1000, clip_dim=512, text_dim=768, is_visual=False):
        self.num_samples = num_samples
        self.is_visual = is_visual
        
        # Fix random seed for reproducible projection matrix
        np.random.seed(42)
        half_samples = num_samples // 2
        
        # Positive pairs
        base_pos = np.random.randn(half_samples, text_dim).astype(np.float32)
        transcript_pos = base_pos + 0.1 * np.random.randn(half_samples, text_dim).astype(np.float32)
        
        if is_visual:
            # For keyframes: positive visual is a projection/correlated representation of base_pos
            # We construct a simple projection to make them semantically correlated
            proj_w = np.random.randn(text_dim, clip_dim).astype(np.float32) / np.sqrt(text_dim)
            visual_pos = np.dot(base_pos, proj_w) + 0.05 * np.random.randn(half_samples, clip_dim).astype(np.float32)
            self.input_a = np.concatenate([visual_pos, visual_pos], axis=0) # Double length for pos and neg
        else:
            # For slide OCR vs transcript
            ocr_pos = base_pos + 0.2 * np.random.randn(half_samples, text_dim).astype(np.float32)
            self.input_a = np.concatenate([ocr_pos, ocr_pos], axis=0)
            
        # Negative pairs: pairing mismatched ones (shift by 1 index)
        transcript_neg = np.roll(transcript_pos, shift=1, axis=0)
        
        self.transcript_embeddings = np.concatenate([transcript_pos, transcript_neg], axis=0)
        
        # Labels: 1.0 for positive, 0.0 for negative
        labels_pos = np.ones(half_samples, dtype=np.float32)
        labels_neg = np.zeros(half_samples, dtype=np.float32)
        self.labels = np.concatenate([labels_pos, labels_neg], axis=0)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "a": torch.tensor(self.input_a[idx], dtype=torch.float32),
            "transcript": torch.tensor(self.transcript_embeddings[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def train_model(model, train_loader, val_loader, num_epochs=20, name="SlideTextMatcher"):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCELoss()
    
    train_losses = []
    val_losses = []
    
    print(f"Training {name}...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            a = batch["a"].to(device)
            t = batch["transcript"].to(device)
            y = batch["label"].to(device)
            
            optimizer.zero_grad()
            pred = model(a, t)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * a.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                a = batch["a"].to(device)
                t = batch["transcript"].to(device)
                y = batch["label"].to(device)
                
                pred = model(a, t)
                loss = criterion(pred, y)
                total_val_loss += loss.item() * a.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"[{name}] Epoch {epoch:02d}/{num_epochs:02d} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    return train_losses, val_losses

def main():
    output_dir = project_root / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = project_root / "storage" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 20
    
    # 1. Train SlideTextMatcher
    full_ds_text = BinaryMatchingDataset(num_samples=1200, is_visual=False)
    train_ds_text, val_ds_text = torch.utils.data.random_split(full_ds_text, [1000, 200])
    train_loader_text = DataLoader(train_ds_text, batch_size=32, shuffle=True)
    val_loader_text = DataLoader(val_ds_text, batch_size=32, shuffle=False)
    
    model_text = SlideTextMatcher().to(device)
    losses_train_text, losses_val_text = train_model(
        model_text, train_loader_text, val_loader_text, num_epochs=num_epochs, name="SlideTextMatcher"
    )
    torch.save(model_text.state_dict(), model_dir / "slide_matcher.pth")
    print(f"Saved slide_matcher.pth\n")
    
    # 2. Train KeyframeMatcher
    full_ds_vis = BinaryMatchingDataset(num_samples=1200, is_visual=True)
    train_ds_vis, val_ds_vis = torch.utils.data.random_split(full_ds_vis, [1000, 200])
    train_loader_vis = DataLoader(train_ds_vis, batch_size=32, shuffle=True)
    val_loader_vis = DataLoader(val_ds_vis, batch_size=32, shuffle=False)
    
    model_vis = KeyframeMatcher().to(device)
    losses_train_vis, losses_val_vis = train_model(
        model_vis, train_loader_vis, val_loader_vis, num_epochs=num_epochs, name="KeyframeMatcher"
    )
    torch.save(model_vis.state_dict(), model_dir / "keyframe_matcher.pth")
    print(f"Saved keyframe_matcher.pth\n")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses_train_text, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), losses_val_text, label="Val Loss", marker="x")
    plt.title("SlideTextMatcher Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), losses_train_vis, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), losses_val_vis, label="Val Loss", marker="x")
    plt.title("KeyframeMatcher Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = output_dir / "binary_loss_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved binary classifiers loss curve plot to: {plot_path}")

if __name__ == "__main__":
    main()
