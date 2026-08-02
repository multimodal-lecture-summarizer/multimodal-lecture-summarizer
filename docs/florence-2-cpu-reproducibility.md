# Florence-2: nguyên nhân sinh text rác và hướng khắc phục

## 1. Tổng quan

Florence-2 trong pipeline thị giác có nhiệm vụ tạo caption cho các keyframe của video.
Trước khi sửa, cùng một model vendored có thể sinh caption đúng trên GPU nhưng tạo chuỗi token
vô nghĩa hoặc text rác khi chạy trên CPU. Việc chỉ đổi `float16` sang `float32` chưa giải quyết
triệt để vì lỗi không xuất phát từ một biến `dtype` duy nhất, mà từ toàn bộ hợp đồng runtime:
phiên bản thư viện, kiểu dữ liệu đầu vào, thuật toán attention, cấu hình giải mã, trạng thái ngẫu
nhiên, mức song song và tính toàn vẹn của model.

Mục tiêu của thay đổi là tạo một đường suy luận Florence-2 có thể tái lập trên Windows và Linux.
CPU được chọn làm chế độ mặc định trên mọi máy, kể cả máy có GPU. CUDA vẫn được hỗ trợ như một
chế độ tăng tốc chủ động, nhưng không được xem là cam kết tuyệt đối giống CPU từng token trên mọi
kiến trúc GPU và phiên bản driver.

## 2. Biểu hiện của lỗi

- Caption chứa các từ không liên quan, ký tự lạ hoặc các đoạn lặp vô nghĩa.
- CPU và GPU có thể trả kết quả khác nhau dù dùng cùng ảnh và cùng checkpoint.
- Đổi model sang `float32` trên CPU vẫn có thể còn lỗi.
- Kết quả phụ thuộc môi trường cài đặt và khó tái hiện trên máy khác.
- Worker có thể khởi động bình thường nhưng chỉ lỗi sau khi đã nhận task nếu file model hoặc
  tokenizer bị thiếu, sai phiên bản hay chưa được tải đầy đủ bằng Git LFS.

## 3. Phân tích nguyên nhân

### 3.1. Phiên bản Transformers không tương thích

Model Florence-2 đang dùng mã Python vendored và được nạp bằng `trust_remote_code=True`.
`transformers==5.14.1` không tương thích với phần mã Florence-2 này. Sự thay đổi API và hành vi
nội bộ có thể làm sai quá trình xử lý token hoặc logits, dẫn đến token đầu ra bị hỏng dù checkpoint
không thay đổi.

Đây là nguyên nhân quan trọng nhất khiến thay đổi `dtype` riêng lẻ không đủ. `dtype` đúng không thể
bù cho một processor, tokenizer hoặc lớp model đang chạy trên phiên bản thư viện không tương thích.

### 3.2. Chọn thiết bị và kiểu dữ liệu ngầm định

Code cũ chọn `float16` nếu `torch.cuda.is_available()` và dùng `self.device` chung của
`SemanticAnalyzer`. Cách này làm hành vi Florence-2 phụ thuộc phần cứng của máy và cấu hình của các
model khác trong pipeline. Ngoài ra, gọi `.to(device, dtype)` trên toàn bộ batch có nguy cơ ép cả
`input_ids` sang kiểu số thực, trong khi token ID phải giữ kiểu số nguyên.

### 3.3. Thuật toán suy luận và giải mã chưa được khóa

Runtime trước đây không khóa implementation của attention, seed và deterministic algorithms.
Lệnh `generate()` chỉ đặt `max_new_tokens` và `num_beams`, nên giới hạn sinh quá dài và thiếu các
ràng buộc chống lặp. Sai số số học nhỏ có thể thay đổi token được chọn ở một bước beam search, rồi
lan thành một chuỗi đầu ra hoàn toàn khác.

### 3.4. Celery chạy nhiều luồng hoặc nhiều tiến trình

Các launcher từng dùng `threads` hoặc concurrency lớn hơn 1. Trong khi đó, seed và các cờ
deterministic của PyTorch, cuDNN và TF32 là trạng thái toàn tiến trình. Hai task Florence-2 chạy xen
kẽ có thể ghi đè trạng thái của nhau, tạo kết quả không ổn định và làm tăng mạnh bộ nhớ do nhiều bản
sao model cùng được nạp.

### 3.5. Phiên bản dependency có thể trôi giữa các máy

Các ràng buộc dạng `>=` cho Torch, NumPy, Transformers, timm hoặc einops cho phép Windows và Linux
cài các phiên bản khác nhau theo thời điểm. Với model dùng custom code, một thay đổi nhỏ trong
Transformers, tokenizer, Torch hoặc torchvision cũng có thể thay đổi hành vi hoặc làm model không
nạp được.

### 3.6. Không kiểm tra đầy đủ tài sản model

Checkpoint Florence-2 không chỉ gồm `model.safetensors`. Kết quả còn phụ thuộc vào config,
processor, tokenizer, vocabulary và ba file Python vendored. Nếu Git LFS chỉ để lại pointer, file bị
thiếu hoặc một file text bị thay đổi, worker cũ không phát hiện trước khi nhận task.

Khác biệt xuống dòng CRLF trên Windows và LF trên Linux cũng cần được xử lý khi so sánh checksum
của file text; nếu không, hai checkout có nội dung logic giống nhau vẫn bị xem là khác nhau.

## 4. Hướng đã giải quyết

### 4.1. Thiết lập hợp đồng runtime cố định

Module `ai_workers/modules/visual_v2/florence_runtime.py` định nghĩa một hợp đồng duy nhất:

- Chỉ hỗ trợ Python 3.10 và 3.11.
- Khóa chính xác các phiên bản trực tiếp ảnh hưởng Florence-2, gồm Torch 2.5.1,
  torchvision 0.20.1, Transformers 4.57.6, tokenizers 0.22.2, NumPy 1.26.4,
  Pillow 12.2.0, timm 1.0.27 và einops 0.8.2.
- Chỉ chấp nhận `FLORENCE_DEVICE=cpu` hoặc `FLORENCE_DEVICE=cuda`.
- Luôn dùng `torch.float32` và attention implementation `eager`.
- CPU là giá trị mặc định, không tự chuyển sang CUDA chỉ vì máy có GPU.

Worker dừng ngay khi khởi động nếu Python, package, thiết bị hoặc cấu hình CUDA không thỏa hợp đồng.

### 4.2. Sửa đường nạp model và tensor

Florence-2 hiện được nạp bằng FP32/eager trên thiết bị đã chọn và chuyển sang `eval()` trước suy
luận. Batch được chuyển thiết bị theo từng tensor; chỉ `pixel_values` được ép sang FP32, còn
`input_ids` giữ nguyên kiểu số nguyên.

Giải mã được cố định với các tham số chính:

```text
do_sample=False
num_beams=3
max_new_tokens=64
early_stopping=True
no_repeat_ngram_size=3
repetition_penalty=1.2
```

Cách này loại bỏ sampling ngẫu nhiên, giới hạn chiều dài và giảm chuỗi lặp vô nghĩa.

### 4.3. Khóa trạng thái deterministic

`FlorenceDeterminism` thực hiện các bước sau quanh toàn bộ lần suy luận:

- Đặt seed cố định cho Python, NumPy, Torch và CUDA.
- Bật `torch.use_deterministic_algorithms(True)`.
- Với CUDA: bật cuDNN deterministic, tắt benchmark và TF32, yêu cầu
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Dùng process-wide lock để không có hai lần suy luận Florence-2 thay đổi trạng thái toàn cục cùng
  lúc.
- Khôi phục toàn bộ RNG và cờ runtime sau khi hoàn tất hoặc khi thiết lập thất bại.

### 4.4. Chuẩn hóa Celery thành một worker tuần tự

`run.bat`, `run_win.ps1`, `run.sh`, Dockerfile và cấu hình Celery đều dùng:

```text
--pool=solo --concurrency=1
```

Thiết lập này nhất quán giữa Windows và Linux, tránh tranh chấp trạng thái deterministic và hạn chế
việc nạp nhiều bản sao model lớn vào RAM hoặc VRAM.

### 4.5. Xác minh checksum toàn bộ model vendored

Manifest SHA-256 bao phủ checkpoint, config, processor, tokenizer, vocabulary và mã Python
Florence-2. Worker xác minh manifest trước khi nhận task. Hash của file text được chuẩn hóa xuống
dòng trước khi tính để cùng một nội dung cho kết quả giống nhau trên Windows và Linux; checkpoint
nhị phân vẫn được hash theo byte gốc.

Nếu checkpoint thiếu, thông báo yêu cầu chạy `git lfs pull`. Nếu asset khác bị thiếu hoặc sai hash,
worker yêu cầu khôi phục checkout đã xác minh thay vì tiếp tục chạy với trạng thái không rõ ràng.

### 4.6. Tách chế độ CPU mặc định và GPU chủ động

Docker mặc định dùng image Python 3.11 trên Debian và không bắt buộc máy có NVIDIA GPU.
`docker-compose.yml` đặt `FLORENCE_DEVICE=cpu`. File `docker-compose.gpu.yml` chỉ là overlay cấp
quyền GPU cho các stage cần tăng tốc.

Khi chủ động đặt `FLORENCE_DEVICE=cuda`, Florence-2 vẫn dùng FP32 và cấu hình deterministic. Tuy
nhiên, CUDA kernel, GPU architecture và driver có thể tạo sai số số học khác CPU. Vì vậy chế độ
được dùng để bảo đảm kết quả đa máy là CPU; CUDA là lựa chọn đánh đổi tính đồng nhất tuyệt đối để
lấy hiệu năng.

## 5. Kiểm thử và bằng chứng xác nhận

Bộ test mới kiểm tra:

- Phiên bản runtime và thiết bị mặc định.
- Từ chối device không hợp lệ hoặc CUDA không khả dụng.
- Phát hiện package bị trôi phiên bản.
- Phát hiện model thiếu hoặc sai checksum.
- Hash file text ổn định giữa CRLF và LF.
- Process-wide lock thực sự tuần tự hóa hai luồng.
- Trạng thái RNG và deterministic được phục hồi cả khi thành công lẫn thất bại.
- Golden inference bằng checkpoint thật với ảnh hình vuông đỏ trên nền trắng.

Golden caption mong đợi là:

```text
a red square on a white background
```

Kết quả này đã được xác nhận trên Windows CPU và Linux CPU trong container Python 3.11. Workflow
`.github/workflows/florence-reproducibility.yml` tiếp tục chạy golden test trên
`windows-latest` và `ubuntu-latest` khi code AI worker thay đổi.

## 6. Hợp đồng vận hành trên máy mới

Trước khi khởi động worker:

```bash
git lfs pull
python -m pip install -r ai_workers/requirements.txt
```

Giữ `FLORENCE_DEVICE=cpu` nếu yêu cầu chính là kết quả đồng nhất giữa máy Windows, Linux, máy chỉ
có CPU và server có GPU. Khi worker khởi động hợp lệ, log phải chứa:

```text
[Startup] Florence-2 runtime verified: cpu/float32/eager, all asset SHA-256 checks OK.
```

Không bỏ qua lỗi startup bằng cách nới phiên bản package hoặc sửa checksum thủ công. Nếu manifest
không khớp, cần tải lại Git LFS hoặc khôi phục đúng model checkout.

## 7. Giới hạn còn lại

- Không thể cam kết toán học rằng CPU và mọi GPU sẽ luôn sinh cùng từng token. Vì vậy CPU là chuẩn
  đối chiếu đa nền tảng.
- Các dependency ngoài đường suy luận Florence-2 vẫn có thể có hợp đồng phiên bản riêng cho
  WhisperX, PaddleOCR hoặc các stage khác.
- Golden test kiểm tra một đầu vào cố định. Nó bảo vệ runtime khỏi drift rõ ràng nhưng không thay
  thế bộ benchmark caption trên tập keyframe thực tế.
- Sau khi cập nhật package hoặc model asset có chủ đích, cần chạy lại golden test trên cả Windows
  và Linux rồi mới cập nhật version hoặc manifest.

## 8. Tham chiếu

- Runtime contract: `ai_workers/modules/visual_v2/florence_runtime.py`
- Đường suy luận: `ai_workers/modules/visual_v2/semantic.py`
- Phiên bản package: `ai_workers/requirements.txt`
- Startup validation: `ai_workers/core/celery_app.py`
- Unit tests: `ai_workers/tests/test_florence_runtime.py`
- Golden test: `ai_workers/tests/test_florence_golden.py`
- CI đa nền tảng: `.github/workflows/florence-reproducibility.yml`
- Cài đặt và vận hành: `docs/setup-guide.md`
