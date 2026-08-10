import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path to import fusion_network
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.models.fusion_network import MultimodalSceneEncoder

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LectureFeatureDataset(Dataset):
    def __init__(self, num_samples=1000, clip_dim=512, text_dim=768):
        self.num_samples = num_samples
        # Generate random baseline features
        base_features = np.random.randn(num_samples, text_dim).astype(np.float32)
        
        # Transcript inherits from base_features with small noise
        self.transcript_embeddings = base_features + 0.1 * np.random.randn(num_samples, text_dim).astype(np.float32)
        # OCR inherits from base_features with moderate noise
        self.ocr_embeddings = base_features + 0.3 * np.random.randn(num_samples, text_dim).astype(np.float32)
        # Visual (CLIP) is random projection mapping
        self.visual_embeddings = np.random.randn(num_samples, clip_dim).astype(np.float32)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "visual": torch.tensor(self.visual_embeddings[idx]),
            "ocr": torch.tensor(self.ocr_embeddings[idx]),
            "transcript": torch.tensor(self.transcript_embeddings[idx])
        }

def contrastive_loss(joint_embeddings, text_embeddings, temp=0.07):
    # L2 normalization
    joint_norm = F.normalize(joint_embeddings, p=2, dim=-1)
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Cosine similarity matrix
    logits = torch.matmul(joint_norm, text_norm.T) / temp
    
    # Targets are diagonal indices
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    
    return (loss_v2t + loss_t2v) / 2

def main():
    # Setup directories
    output_dir = project_root / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = project_root / "storage" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize datasets & loaders
    train_dataset = LectureFeatureDataset(num_samples=800)
    val_dataset = LectureFeatureDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = MultimodalSceneEncoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    num_epochs = 20
    train_losses = []
    val_losses = []
    
    print("Starting training Multimodal Scene Encoder...")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            v = batch["visual"].to(device)
            o = batch["ocr"].to(device)
            t = batch["transcript"].to(device)
            
            optimizer.zero_grad()
            joint_emb = model(v, o, t)
            proj_t = model.norm_t(model.proj_transcript(t))
            loss = contrastive_loss(joint_emb, proj_t)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * v.size(0)
            
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # Validation evaluation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                v = batch["visual"].to(device)
                o = batch["ocr"].to(device)
                t = batch["transcript"].to(device)
                
                joint_emb = model(v, o, t)
                proj_t = model.norm_t(model.proj_transcript(t))
                loss = contrastive_loss(joint_emb, proj_t)
                total_val_loss += loss.item() * v.size(0)
                
        avg_val_loss = total_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    print("-" * 60)
    print("Training complete!")
    
    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss", marker="x")
    plt.title("Multimodal Scene Encoder - Contrastive Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = output_dir / "hybrid_loss_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss curve plot to: {plot_path}")
    
    # Save weights
    model_path = model_dir / "scene_encoder.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")

if __name__ == "__main__":
    main()
