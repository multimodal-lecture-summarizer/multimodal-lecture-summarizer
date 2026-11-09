# Research: video YouTube phù hợp pipeline hiện tại

Thời điểm: 2026-11-08 19:32 (Asia/Bangkok), cập nhật 19:49 theo kết quả trong `benchmark/`, 20:07 thêm nhóm D từ MLPDataset. Phạm vi: chỉ URL YouTube. Người dùng tự chạy, nên không tải media.

## Tóm tắt

**Kết luận: benchmark đủ để nói hệ thống phù hợp với nhóm A ở phần lời nói và tóm tắt văn bản, chưa đủ để nói toàn hệ thống làm tốt.**

- Nhóm A khớp phạm vi đã chốt ngày 03/11: bài giảng hoặc báo cáo hội nghị tiếng Anh, một diễn giả, slide có chữ.
- **Đã chạy trên pipeline:** duy nhất 1 video YouTube (`FLWvcdfGPxs`) có kết quả đầy đủ cả OCR.
- **Chưa chạy:** 29 video còn lại. Với chúng chỉ biết: link sống, 360p đạt chuẩn kỹ thuật, tiếng Anh theo metadata. Chưa xem hình.
- Chuyển slide, caption, hỏi đáp và toàn pipeline chưa có kết quả benchmark. Phần này chỉ có thể kết luận sau khi chạy thật.
- **Không có video YouTube nào được nghiên cứu trên mọi tầng**, nên chỉ chọn được theo từng tầng: nhóm A cho ASR và tóm tắt, nhóm D cho chuyển slide.

## Kết luận theo benchmark

| Tầng | Kết quả benchmark | Dùng làm lý do | Giới hạn phải nói |
|---|---|---|---|
| ASR | Qwen3-ASR 0.6B được chọn trên TED-LIUM: WER validation 3,91%, Whisper Base 6,64%, thấp nhất trong 6 hệ thống. Native test 3,53% chỉ là diagnostic | **Có** | Đánh giá trên đoạn ngắn, app chạy cả bài. Không đo thuật ngữ chuyên ngành. Chưa so với Whisper `small` |
| Tóm tắt văn bản | LED `confirmed` trên TIB (519 ID): ROUGE-L hơn BART +0,0097 [+0,0040; +0,0153], hơn LexRank +0,0340 [+0,0286; +0,0394] | **Có, chỉ tầng văn bản** | Hơn BART dưới ngưỡng thực tế 0,01. MiniCheck post-hoc: LED kém trung thực hơn BART (cảnh báo 26,4% so với 10,0%, nhưng hand check 5/11 câu của LED là cảnh báo sai). ROUGE-L đo độ khớp với abstract, không đo tính đúng sự thật |
| OCR | Crop IIIT-CVID: PaddleOCR CER 11,72%, EasyOCR 24,44%. Spotting R5: Paddle DBv3+Recv4 Hmean 6,80% | **Một phần, chỉ phần nhận dạng** | Nhãn `no_selection`, chưa đo riêng trên slide văn bản. Hmean thấp có phần do Paddle gộp nhiều từ (40,99% vùng) trong khi nhãn theo từng từ |
| Chuyển slide | Chưa có kết quả. MLPDataset chưa đủ điều kiện chấm chính thức | **Không** | Video mốc có cảnh dài 459/743s |
| Caption | Chưa có ground truth | **Không** | Giá trị với slide chữ chưa kiểm chứng |
| Detection, tracking | COCO, LVVO, DanceTrack; đã ngoài phạm vi từ 03/11 | **Không** | Không dùng để biện minh cho video bài giảng |
| Hỏi đáp, toàn pipeline | Chưa đánh giá | **Không** | RQ3 đang HOLD, phase tích hợp chưa chạy |

Khác biệt giữa benchmark và app: LED được đo trên transcript sẵn có, không có OCR. App dùng transcript từ ASR và mặc định xen chữ OCR (`LED_INCLUDE_OCR=true`), phần này chưa có số đo.

Câu dùng khi trình bày:

> Nhóm video được chọn thuộc đúng phạm vi nghiên cứu: báo cáo học thuật tiếng Anh, một diễn giả, slide có chữ. Ở phần lời nói, Qwen3-ASR 0.6B đạt WER 3,91% trên TED-LIUM, thấp nhất trong sáu hệ thống. Ở phần tóm tắt văn bản, LED được xác nhận trên 519 video TIB theo ROUGE-L, nhưng mức hơn BART nhỏ và độ trung thực thấp hơn BART. PaddleOCR có CER 11,72% trên crop bài giảng IIIT-CVID, chưa được đo riêng trên slide văn bản. Chuyển slide, hỏi đáp và toàn pipeline chưa có kết quả benchmark, nên phần này sẽ được đánh giá bằng lần chạy thực tế trên các video trên.

## Bằng chứng từ lần chạy thật (Postgres container, chỉ đọc)

17 video, 12 job xong, thực chất 2 video khác nhau. 4 job lỗi đều là bot-check của YouTube khi tải, không liên quan nội dung.

| Video | Cảnh | OCR | Cảnh dài nhất | Tóm tắt LED | Nhận xét |
|---|---|---|---|---|---|
| OSDI '23 `FLWvcdfGPxs`, 743s, YouTube, lần chạy mới nhất | 8 | 7/8 cảnh có chữ, tin cậy TB từng cảnh 0,908–0,974 | 459s | 513 ký tự, mạch lạc, đúng ý chính | Đọc được tiêu đề slide, nhãn biểu đồ. Bỏ sót nhiều chuyển slide |
| Google "Date - Data - Strikes", 831s, 720p, upload | 12–15 | 0 (không kết luận được) | n/a | 524 ký tự, mạch lạc | Caption gần như chỉ "man in gray shirt" |

Lưu ý khi đọc số liệu:
- 3 lần chạy OSDI trước và mọi lần chạy video Google đều OCR = 0, kể cả cùng video OSDI. Lần chạy có OCR là lần muộn nhất. Vì vậy OCR = 0 ở video Google không chứng minh video đó không có chữ.
- Lỗi ASR quan sát được: "Delegated" thành "dedicated", "DuVisor" thành "divisor", "IPADS" thành "IPad". Có đoạn câu lặp lại trong transcript ("The attack effects are also severe Over 90 of"), chưa rõ do người nói hay do ghép đoạn ASR.
- Tiêu đề, chapter, takeaways không do LED sinh. Chapter ra "Chapter 1/2/3", takeaways là câu trích. Mọi video sẽ có cùng hạn chế này.

## Tiêu chí rút ra

Điều kiện cứng (code, `video_standards`, kiểm ở Stage 0, sai là job fail):
- ≤3600s, ≤500 MB, h264/hevc, rộng ≥640 và cao ≥360.
- YouTube qua `player_client` hiện tại: 71/71 video chỉ ra format 18 (360p). Vì vậy video phải đúng 16:9. 640x312 bị từ chối (`X0TVmDDmDWY`); 4:3 ra 480x360 và bị từ chối theo cùng quy tắc. Trên MLPDataset, 245/324 nguồn bị loại vì độ phân giải, 115 nguồn là đúng 480x360.

Điều kiện chất lượng:
- Tiếng Anh: Qwen3-ASR, LED, `PADDLEOCR_LANG=en` đều theo phạm vi tiếng Anh.
- Có slide chữ lớn: 360p vẫn đọc tốt tiêu đề và nhãn lớn trên talk OSDI.
- Nội dung học thuật: tầng văn bản của LED được xác nhận trên TIB, là video học thuật. 2 mẫu chạy thật đều ra tóm tắt mạch lạc, n quá nhỏ để so.
- Tránh video chỉ quay người nói: caption không mang thông tin.
- ContentDetector chấm thay đổi toàn khung hình, ngưỡng 27. Slide đổi trong bố cục cố định dễ bị bỏ sót. Hạn chế này áp dụng cho mọi nhóm và benchmark chưa đo.

## Danh sách

### Mốc so sánh (đã chạy thật)

| # | URL | Video | Giây | MB (360p) | Phụ đề người viết | Chapter YouTube |
|---|---|---|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=FLWvcdfGPxs | OSDI '23 - Security and Performance in the Delegated User-level Virtualization | 743 | 18.3 | không | 9 |

### Nhóm A: khớp phạm vi và thể loại TIB, cùng kênh với mốc (chưa chạy)

Nhóm duy nhất nên dùng cho luận điểm "phù hợp". Thứ tự: OSDI '23 trước, rồi NSDI, FAST, ATC, Security cùng năm 2023. Trong mỗi hội nghị, ưu tiên nhiều chapter. Chỉ lấy video có chapter để có mốc đối chiếu chia đoạn. Không biết chapter do người đặt hay YouTube tự sinh.

| # | URL | Video | Giây | MB (360p) | Phụ đề người viết | Chapter YouTube |
|---|---|---|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=MfSp7WJuz-8 | OSDI '23 - Core slicing: closing the gap between leaky confidential VMs and bare-metal cloud | 883 | 19.4 | không | 12 |
| 2 | https://www.youtube.com/watch?v=QY-83DX-R2I | OSDI '23 - Take Out the TraChe: Maximizing (Tra)nsactional Ca(che) Hit Rate | 892 | 19.0 | không | 9 |
| 3 | https://www.youtube.com/watch?v=OFjlUqHbrMQ | OSDI '23 - Cilantro: Performance-Aware Resource Allocation for General Objectives via Online... | 949 | 21.6 | không | 7 |
| 4 | https://www.youtube.com/watch?v=tnkmpsAHyMg | OSDI '23 - ExoFlow: A Universal Workflow System for Exactly-Once DAGs | 963 | 22.6 | không | 6 |
| 5 | https://www.youtube.com/watch?v=snKg5vIYjb4 | OSDI '23 - Characterizing Off-path SmartNIC for Accelerating Distributed Systems | 878 | 21.1 | không | 6 |
| 6 | https://www.youtube.com/watch?v=oyumGtYN558 | OSDI '23 - ShRing: Networking with Shared Receive Rings | 858 | 18.7 | không | 6 |
| 7 | https://www.youtube.com/watch?v=X929vAkovX4 | NSDI '23 - LOCA: A Location-Oblivious Cellular Architecture | 979 | 22.3 | không | 12 |
| 8 | https://www.youtube.com/watch?v=QOkDuS9enUA | NSDI '23 - DChannel: Accelerating Mobile Applications With Parallel High-bandwidth... | 1147 | 25.4 | không | 11 |
| 9 | https://www.youtube.com/watch?v=u3YG5hRQjdA | NSDI '23 - Hermit: Low-Latency, High-Throughput, and Transparent Remote Memory... | 1063 | 23.6 | không | 9 |
| 10 | https://www.youtube.com/watch?v=DcDnrZY11Yw | FAST '23 - Perseus: A Fail-Slow Detection Framework for Cloud Storage Systems | 971 | 22.2 | không | 9 |
| 11 | https://www.youtube.com/watch?v=zGnn7IkIdiE | FAST '23 - Fast Application Launch on Personal Computing/Communication Devices | 1086 | 30.4 | không | 9 |
| 12 | https://www.youtube.com/watch?v=RfAq1JbHbXI | USENIX ATC '23 - EnvPipe: Performance-preserving DNN Training Framework for Saving Energy | 1250 | 30.5 | không | 9 |
| 13 | https://www.youtube.com/watch?v=HVRRL9Ky6Eo | USENIX Security '23 - Ethical Frameworks and Computer Security Trolley Problems: Foundations for... | 818 | 18.4 | không | 7 |

### Nhóm B: trong phạm vi nhưng chưa chắc (chưa chạy)

Bài giảng và talk kỹ thuật đúng phạm vi, nhưng chưa rõ TIB và TED-LIUM đại diện được giọng hay dạng trình bày của chúng. Video có phụ đề người viết cho phép tính WER thật. `II4giR4vOOo` dài 3376s, gần trần 3600s. Chưa xác nhận hình có slide chữ hay không.

| # | URL | Video | Giây | MB (360p) | Phụ đề người viết | Chapter YouTube |
|---|---|---|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=tA42nHmmEKw | #1 Introduction to Python for Data Science / Python for Data Science | 961 | 32.1 | có | 6 |
| 2 | https://www.youtube.com/watch?v=fC7V8QsPBec | Week 1 - Lecture 1 - Introduction to Machine Learning | 927 | 37.6 | có | 5 |
| 3 | https://www.youtube.com/watch?v=XjOnY-OLAPc | Quicksort, Timsort, Powersort - PyCon US 2023 talk | 1768 | 46.3 | không | 11 |
| 4 | https://www.youtube.com/watch?v=II4giR4vOOo | MIT Introduction to Deep Learning / 6.S191 | 3376 | 67.5 | không | 11 |

### Nhóm C: ngoài phạm vi, chỉ dùng làm phản ví dụ (chưa chạy)

Video giải thích hoặc khóa học ngắn. Benchmark không hỗ trợ, không dùng cho luận điểm "phù hợp".

| # | URL | Video | Giây | MB (360p) | Phụ đề người viết | Chapter YouTube |
|---|---|---|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=CqOfi41LfDw | The Essential Main Ideas of Neural Networks | 1134 | 22.9 | có | 6 |
| 2 | https://www.youtube.com/watch?v=yIYKR4sgzI8 | StatQuest: Logistic Regression | 527 | 9.7 | có | 5 |
| 3 | https://www.youtube.com/watch?v=p-ltr1C7u2o | #31 Machine Learning Specialization [Course 1, Week 3, Lesson 1] | 588 | 13.0 | không | 4 |

### Nhóm D: có mốc chuyển slide từ MLPDataset (chưa chạy)

Dùng để đo recall chuyển slide, tầng benchmark đang bỏ trống. Dataset nào chọn được video YouTube cụ thể:

| Dataset | Chọn được không | Lý do |
|---|---|---|
| TIB (tóm tắt) | Không | Nguồn là TIB AV-Portal, kết quả có ý nghĩa ở mức cả tập |
| TED-LIUM (ASR) | Không | Bài nói không slide; kết quả chi tiết từng đoạn không còn đủ ở máy |
| IIIT-CVID (OCR) | Không | Không có ánh xạ video sang URL; 1/7 khóa test là slide |
| MLPDataset (chuyển slide) | Có | ID là ID YouTube, có mốc theo từng video |

Đối chiếu điều kiện cứng (metadata, 326 nguồn trong ledger, 324 đọc được, 2 lỗi format khi đọc):
- 42/324 đạt. Loại vì độ phân giải 245, thời lượng 62, YouTube ghi ngôn ngữ khác tiếng Anh 23 (một nguồn có thể bị nhiều lý do).
- 13 video có điểm chuyển slide đã duyệt tay đều không đạt điều kiện cứng. 42 video đạt chỉ có mốc gốc của dataset, chưa duyệt.
- Mốc gốc lẫn thay đổi trong slide và phần phụ; `formal_scoring_eligible=false`, chỉ dùng mức chẩn đoán.
- 34/35 điểm đã duyệt là hard cut, đúng loại ContentDetector nhắm tới. Mẫu chọn có chủ đích, không phải ước lượng tỷ lệ.

Quy tắc chọn: đạt điều kiện cứng, YouTube ghi `en`, tiêu đề không phải bài luyện đề (quiz, practice), lấy 3 video có mật độ mốc cao nhất ở mỗi nhóm `anat-1`, `bio-1`, `dental`. Bỏ `speaking` vì đa số không có nhãn ngôn ngữ, bỏ `anat-2` vì chỉ còn 1 video không phải quiz. Mật độ cao nghĩa là nhiều điểm kiểm tra, không có nghĩa kết quả tốt hơn. Chưa xem hình.

| # | URL | Video | Nhóm | Giây | Số mốc | Mốc/phút |
|---|---|---|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=IotJiLgIwvQ | Anatomy and Physiology Chapter 10 Part D Lecture | anat-1 | 2259 | 55 | 1,46 |
| 2 | https://www.youtube.com/watch?v=2U3DsK7BHVU | Anatomy & Physiology Chapter 9 Part B Lecture | anat-1 | 1601 | 38 | 1,42 |
| 3 | https://www.youtube.com/watch?v=RH-Pe9MBDBU | Anatomy & Physiology Chapter 8 Lecture Part A | anat-1 | 2903 | 56 | 1,16 |
| 4 | https://www.youtube.com/watch?v=uM1JOFaATP4 | Chapter 32 Tissues and Endocrine System | bio-1 | 3397 | 53 | 0,94 |
| 5 | https://www.youtube.com/watch?v=g1JIbOMWBBM | Chapter 27 The Rise of the Invertebrate Animals | bio-1 | 3431 | 52 | 0,91 |
| 6 | https://www.youtube.com/watch?v=pUH2_SGnm9g | Chapter 39 Muscoloskeletal System (1 mốc đang HOLD) | bio-1 | 3053 | 45 | 0,88 |
| 7 | https://www.youtube.com/watch?v=hPmpU72Cgsg | Head & Neck Anatomy, Pharynx & Larynx Muscles | dental | 432 | 10 | 1,39 |
| 8 | https://www.youtube.com/watch?v=xcyUm5ZEzng | Head & Neck Anatomy, Craniofacial Veins | dental | 562 | 10 | 1,07 |
| 9 | https://www.youtube.com/watch?v=Yeq6LO3q4Vo | Periodontics, New Classification System | dental | 2071 | 32 | 0,93 |

## Đã loại, kèm lý do

| Video | Lý do |
|---|---|
| `X0TVmDDmDWY` | 640x312, bị từ chối ở Stage 0 |
| `CFN5zHzEuGY`, `xuTiAW0OR40`, `_rEXryrChwo` | YouTube ghi ngôn ngữ `ar` |
| `kPWk2ufDuEk`, `2jlaS8j5zqI`, `z7SVieA2VwI` | Podcast, phỏng vấn, trò chuyện trực tiếp theo tiêu đề; không phải bài giảng |
| `eZwHvBsoPn4`, `QpBTM0GO6xI`, `ixRanV-rdAQ`, `H0puaEKcxyE`, `khiWocrQRsc`, `PiRvyjwKwmY` | Khai mạc, keynote sản phẩm, hướng dẫn thi, không có nội dung bài giảng theo tiêu đề |
| `AuNfxVLdo0A` | Keynote 3139s, đạt chuẩn nhưng tốn thời gian; không thêm thông tin so với talk ngắn |

Các video đạt chuẩn khác của USENIX trong mẫu không vào danh sách vì không có chapter hoặc đã đủ đại diện, không phải vì kém hơn.

## Cách chạy và đọc kết quả

- Đặt ngôn ngữ nguồn `en`, ngôn ngữ tóm tắt `en`, cần worker GPU để dùng LED.
- Mỗi link nên ghi 3 số so với mốc: tỷ lệ cảnh có OCR (mốc 7/8), độ tin cậy OCR (mốc 0,91–0,97), cảnh dài nhất so với thời lượng (mốc 459/743s).
- Video có phụ đề người viết: tính thêm WER so với phụ đề đó.
- Nhóm A: tính ROUGE-L so với abstract bài báo trên usenix.org, cùng cách đo với TIB. Nhóm A không có phụ đề người viết, muốn có WER phải gõ tay một mẫu ngắn.
- Nhóm D: so điểm cắt cảnh với mốc MLPDataset theo collar 0,5s của protocol, báo cáo là chẩn đoán.
- Muốn đọc chữ nhỏ trên slide thì upload file độ phân giải cao thay vì dán link.

## Nguồn

- Benchmark: [BENCHMARK.md](../../../benchmark/docs/BENCHMARK.md), [báo cáo ASR](../../../benchmark/docs/results/asr-architecture-evaluation-2026-10-05.md), [DATASET_EDA.md](../../../benchmark/docs/DATASET_EDA.md), [quyết định phạm vi 03/11](../../../plans/reports/protocol-decision-261103-1015-project-scope-english-single-speaker-slides-report.md), [đánh giá MLPDataset](../../../plans/reports/dataset-adjudication-261019-1504-mlpdataset-final-review-report.md).
- MLPDataset: [đối chiếu đã duyệt](../../../benchmark/experiments/registry/datasets/mlpdataset-targeted-adjudication-v2.json), `benchmark/experiments/outputs/dataset_qualification_261010/mlpdataset-boundary-ledger.csv`, [protocol](../../../docs/mlpdataset-slide-transition-protocol.md).
- Dữ liệu nội bộ: Postgres `video_summarizer` trong container all-in-one; `ai_workers/core/youtube_ingest.py`; `backend/app/services/video_inspector.py`; `ai_workers/modules/visual_v2/scene_detector.py`.
- Metadata YouTube: yt-dlp 2026.07.04, không tải media, ngày 2026-11-08, 71 video, 0 lỗi.
- [PySceneDetect API](https://www.scenedetect.com/docs/0.6.6/api.html): ContentDetector dựa trên thay đổi HSV giữa các khung.
- [Lecture slide transition paper](https://ar5iv.labs.arxiv.org/html/2202.03540): độ đa dạng của video bài giảng làm khó phát hiện chuyển slide bằng đặc trưng cấp thấp.

## Câu hỏi còn mở

- Nhóm A có giữ được chất lượng như mốc không? Chỉ có kết quả chạy mới trả lời được.
- LED hay BART cho tầng văn bản vẫn là quyết định mở theo benchmark; app hiện dùng LED theo quyết định sản phẩm.
- Ngưỡng đạt cho từng chỉ số chưa chốt, nên chốt trước khi chạy.
- Có hạ chuẩn rộng tối thiểu 640 hoặc bổ sung PO Token không? Hiện 75% nguồn MLPDataset bị loại vì độ phân giải.
- Vì sao chỉ lần chạy muộn nhất có OCR? Theo tài liệu dự án, video xử lý trước migration không có dữ liệu OCR chi tiết, nhưng `ocr_text` cũng trống, chưa xác định nguyên nhân.
