# Hướng dẫn Thiết kế & Huấn luyện Mô hình Hybrid (Multimodal Scene Encoder)

Tài liệu này cung cấp hướng dẫn kỹ thuật chi tiết để xây dựng, huấn luyện và tích hợp mạng **Multimodal Scene Encoder** sử dụng cơ chế **Cross-modal Attention** (Dung hợp đặc trưng hình ảnh Keyframe, văn bản OCR và thoại Transcript). Đây là phần đóng góp khoa học (novelty) cốt lõi trong luận văn của bạn.

---

## 1. Ý tưởng Kiến trúc & Thuật toán

Mô hình Hybrid được thiết kế theo nguyên lý:
- **Trích xuất đặc trưng tĩnh (Frozen Backbones):** Sử dụng các mô hình pre-trained SOTA để trích xuất đặc trưng của từng phân đoạn (Scene), tránh việc huấn luyện lại các mô hình nền tảng khổng lồ.
- **Dung hợp động (Custom Trainable Fusion Layer):** Thiết kế mạng Transformer-based Fusion học cách liên kết thông tin giữa các modalities (thị giác và ngôn ngữ).

### Luồng xử lý toán học:
Cho một scene $S_i$ trong video, ta có 3 đầu vào đặc trưng:
1. **Visual vector ($v_i$):** Đặc trưng hình ảnh keyframe trích xuất từ CLIP ViT-B/32, chiều $D_v = 512$.
2. **OCR vector ($o_i$):** Đặc trưng văn bản trên slide trích xuất từ PhoBERT/RoBERTa, chiều $D_t = 768$.
3. **Transcript vector ($t_i$):** Đặc trưng đoạn thoại tương ứng trích xuất từ PhoBERT/RoBERTa, chiều $D_t = 768$.

### Các bước tính toán trong Encoder:
1. **Phép chiếu tuyến tính (Projection Layer):** Đưa tất cả các vector về cùng số chiều latent $d_{\text{model}}$ (ví dụ 256):
   $$V'_i = W_v v_i + b_v \quad (V'_i \in \mathbb{R}^{d_{\text{model}}})$$
   $$O'_i = W_o o_i + b_o \quad (O'_i \in \mathbb{R}^{d_{\text{model}}})$$
   $$T'_i = W_t t_i + b_t \quad (T'_i \in \mathbb{R}^{d_{\text{model}}})$$

2. **Căn chỉnh chéo bằng Cross-modal Attention:**
   Cho phép hình ảnh (Visual) "quan sát" và căn chỉnh với văn bản (Transcript + OCR). Ta coi $V'_i$ đóng vai trò là **Query (Q)**, còn $T'_i$ và $O'_i$ (hoặc tổ hợp của chúng) đóng vai trò là **Key (K)** và **Value (V)**:
   $$\text{Context}_i = \text{Attention}(Q=V'_i, K=\text{Concat}(T'_i, O'_i), V=\text{Concat}(T'_i, O'_i))$$
   Điều này trả lời câu hỏi: *"Phần hình ảnh này tương ứng nhất với những từ khóa nào trong lời giảng và slide?"*.

3. **Gộp đặc trưng (Fusion Layer):**
   Kết hợp vector ngữ cảnh vừa học được với vector visual gốc bằng một mạng Feed-Forward (FFN) kết hợp Residual Connection:
   $$z_i = \text{LayerNorm}(V'_i + \text{Context}_i)$$
   $$\text{Scene\_Embedding}_i = \text{FFN}(z_i)$$

---

## 2. Mã nguồn PyTorch tham khảo (Reference Implementation)

Dưới đây là thiết kế class PyTorch hoàn chỉnh cho mô hình **MultimodalSceneEncoder**. Bạn có thể đặt file này tại `experiments/models/fusion_network.py` để chạy thử nghiệm.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalSceneEncoder(nn.Module):
    def __init__(
        self, 
        clip_dim: int = 512,      # Chiều của CLIP ViT keyframe embedding
        text_dim: int = 768,      # Chiều của PhoBERT / SBERT embedding (OCR & Transcript)
        d_model: int = 256,       # Chiều không gian latent chung
        nhead: int = 4,           # Số đầu Attention
        dropout: float = 0.1
    ):
        super(MultimodalSceneEncoder, self).__init__()
        
        # 1. Các lớp chiếu tuyến tính đưa về cùng chiều d_model
        self.proj_visual = nn.Linear(clip_dim, d_model)
        self.proj_ocr = nn.Linear(text_dim, d_model)
        self.proj_transcript = nn.Linear(text_dim, d_model)
        
        # Lớp chuẩn hóa
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        
        # 2. Khối Multi-head Cross-modal Attention
        # Query: Visual (Keyframe)
        # Key/Value: Textual (OCR + Transcript)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        
        # 3. Lớp tổng hợp Feed-Forward (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm_final = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        visual_emb: torch.Tensor,       # Shape: [batch_size, clip_dim]
        ocr_emb: torch.Tensor,          # Shape: [batch_size, text_dim]
        transcript_emb: torch.Tensor    # Shape: [batch_size, text_dim]
    ) -> torch.Tensor:
        """
        Đầu vào:
            visual_emb: Đặc trưng visual của keyframe
            ocr_emb: Đặc trưng OCR của slide
            transcript_emb: Đặc trưng transcript của scene
        Đầu ra:
            scene_joint_emb: Vector biểu diễn dung hợp đa phương thức [batch_size, d_model]
        """
        # Phép chiếu (Projection)
        v = self.norm_v(self.proj_visual(visual_emb))      # [B, d_model]
        o = self.norm_o(self.proj_ocr(ocr_emb))            # [B, d_model]
        t = self.norm_t(self.proj_transcript(transcript_emb))  # [B, d_model]
        
        # Thêm chiều sequence để đưa vào Attention layer
        # sequence_len = 1 cho visual (chỉ có 1 keyframe)
        # sequence_len = 2 cho văn bản (OCR + Transcript)
        q = v.unsqueeze(1) # [B, 1, d_model]
        
        # Gộp OCR và Transcript làm khóa (Key/Value)
        k = torch.stack([o, t], dim=1) # [B, 2, d_model]
        v_val = k # [B, 2, d_model]
        
        # Tính toán Cross-modal Attention
        # Visual chú ý đến thông tin thoại và text trên slide
        attn_output, _ = self.cross_attn(query=q, key=k, value=v_val) # [B, 1, d_model]
        attn_output = attn_output.squeeze(1) # [B, d_model]
        
        # Residual Connection & Normalization
        x = self.norm_final(v.squeeze(1) + attn_output)
        
        # Feed-forward network
        scene_joint_emb = x + self.ffn(x)
        
        return scene_joint_emb # [B, d_model]
```

---

## 3. Quy trình Huấn luyện Tự Giám Sát (Self-supervised Training)

Vì không có nhãn giám sát (supervised labels) về "mức độ dung hợp", bạn nên huấn luyện mô hình này theo phương pháp **Học tương phản (Contrastive Learning - InfoNCE Loss)**.

### Triết lý huấn luyện tương phản:
Một bản ghi dữ liệu huấn luyện gồm cặp $(v_i, t_i)$ ứng với visual keyframe và transcript của cùng một scene $i$. 
- Mô hình phải đưa vector $Scene\_Embedding(v_i, o_i, t_i)$ về gần với vector biểu diễn văn bản gốc $t_i$.
- Đồng thời đẩy nó ra xa vector transcript của các scene khác $t_j$ ($j \neq i$) trong cùng một batch.

### Hàm Loss (InfoNCE):
$$\mathcal{L}_i = -\log \frac{\exp(\cos(\mathbf{z}_i, \mathbf{t}_i) / \tau)}{\sum_{j} \exp(\cos(\mathbf{z}_i, \mathbf{t}_j) / \tau)}$$
Trong đó $\mathbf{z}_i$ là joint embedding sinh ra từ encoder, $\mathbf{t}_i$ là transcript embedding, và $\tau$ là tham số nhiệt độ (temperature scale).

### Code vòng lặp huấn luyện chính (Training Loop):

```python
import torch.optim as optim

def contrastive_loss(joint_embeddings, text_embeddings, temp=0.07):
    # Chuẩn hóa L2
    joint_norm = F.normalize(joint_embeddings, p=2, dim=-1)
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Tính ma trận tương đồng similarity matrix
    logits = torch.matmul(joint_norm, text_norm.T) / temp # [batch_size, batch_size]
    
    # Nhãn đúng nằm trên đường chéo (i, i)
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2

# Khởi tạo mô hình & optimizer
model = MultimodalSceneEncoder().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# Vòng lặp epoch giả lập
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        # Load embeddings đã trích xuất sẵn từ cache
        v_features = batch["visual"].cuda()      # [B, 512]
        o_features = batch["ocr"].cuda()         # [B, 768]
        t_features = batch["transcript"].cuda()  # [B, 768]
        
        optimizer.zero_grad()
        
        # Forward pass
        joint_emb = model(v_features, o_features, t_features)
        
        # Tính Loss tương phản giữa Joint Embedding và Transcript Embedding
        loss = contrastive_loss(joint_emb, t_features)
        
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## 4. Cách Tích hợp vào Codebase Hiện tại

Hiện tại, module `TimelineBuilder` trong file [ai_workers/modules/fusion/timeline.py](file:///c:/Users/admin/multimodal-lecture-summarizer/ai_workers/modules/fusion/timeline.py) đang ở dạng stub. Bạn sẽ tích hợp mô hình Hybrid này tại đây:

1. **Lưu trữ mô hình đã huấn luyện:** Lưu trọng số mô hình dạng `.pth` (ví dụ `storage/models/scene_encoder.pth`).
2. **Khởi tạo và load trọng số:**
   ```python
   # inside ai_workers/modules/fusion/timeline.py
   import torch
   from experiments.models.fusion_network import MultimodalSceneEncoder

   class TimelineBuilder:
       def __init__(self, config: dict | None = None):
           self.config = config or {}
           self.device = "cuda" if torch.cuda.is_available() else "cpu"
           self.encoder = MultimodalSceneEncoder().to(self.device)
           
           # Load trọng số đã pre-train
           model_path = self.config.get("model_path", "storage/models/scene_encoder.pth")
           if os.path.exists(model_path):
               self.encoder.load_state_dict(torch.load(model_path, map_location=self.device))
               self.encoder.eval()
   ```

3. **Chạy suy luận sinh Scene Embedding:**
   Tại hàm `align_modalities`, sau khi nhận được dữ liệu thô, convert chúng sang PyTorch tensors và gọi `self.encoder(v, o, t)`. Kết quả thu được sẽ được lưu vào cơ sở dữ liệu vector ChromaDB hoặc chuyển tiếp tới LLM dưới dạng ngữ cảnh biểu diễn chất lượng cao.

---

## 5. Kết luận & Tác động của mô hình Hybrid

- **Tính khả thi:** Cực kỳ cao vì bạn **chỉ huấn luyện lớp Fusion** mỏng (~2-3 triệu tham số), thời gian huấn luyện chỉ mất từ 30 phút đến vài tiếng trên 1 GPU RTX 3060/4070 thông thường.
- **Tính học thuật:** Đạt chuẩn nghiên cứu khoa học tốt nghiệp. Thử nghiệm thay đổi số đầu attention (`nhead`), thêm/bớt modality (OCR, Visual hoặc Transcript) để đo lường độ sụt giảm chất lượng (Ablation Study) sẽ giúp viết phần "Kết quả nghiên cứu" trong luận văn cực kỳ thuyết phục.
