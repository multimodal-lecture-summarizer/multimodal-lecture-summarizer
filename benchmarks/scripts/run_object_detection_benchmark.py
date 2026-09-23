"""
RQ-OD: Object Detection Benchmark (Bổ sung theo Đề cương)
============================================================
Đánh giá nhiều model object detection nhẹ để chọn model tối ưu cho hệ thống
multimodal lecture summarizer + multi-object video summarization.

Mục tiêu (đề cương mục 2 + 3.1):
- Phát hiện & theo dõi đối tượng (Object Detection)
- Metrics chuẩn: Precision, Recall, mAP@0.5

Phạm vi (theo user): Detection-only (lite), song song với Florence-2 caption.
Constraint: Tesla T4 16GB, pytorch 2.5.1+cu124, transformers 4.57.6.

Candidates evaluated:
  OD-A: YOLOv8n (Ultralytics)        - nano, COCO-pretrained, mature
  OD-B: YOLOv11n (Ultralytics)       - latest nano, improved arch
  OD-C: RT-DETR-r18 (Ultralytics)    - DETR-based, real-time
  OD-D: Florence-2 (Phrase Grounding)- đã có sẵn trong repo, dùng task
                                       OD/REGION_PROPOSAL với text prompt
  OD-E: GroundingDINO-base           - open-vocabulary (optional, heavy)

Outputs: reports/rq_od_benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_VIDEOS_DIR = REPO_ROOT / "experiments" / "datasets" / "eduviqa" / "Videos"
RESULTS_DIR = REPO_ROOT / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models đối sánh — kèm metadata để quyết định dựa trên dữ liệu, không phải cảm tính.
DETECTION_CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": "OD-A",
        "name": "YOLOv8n (Ultralytics)",
        "framework": "ultralytics",
        "model_id": "yolov8n.pt",
        "model_size_mb_estimated": 6.2,
        "pretrained_classes": 80,   # COCO
        "vram_min_mb": 1024,
        "expected_fps_t4": 65.0,   # từ Ultralytics T4 benchmarks
        "open_vocab": False,
        "rationale": "Mature, ổn định, cộng đồng lớn, COCO classes đủ dùng.",
    },
    {
        "id": "OD-B",
        "name": "YOLOv11n (Ultralytics)",
        "framework": "ultralytics",
        "model_id": "yolo11n.pt",
        "model_size_mb_estimated": 5.4,
        "pretrained_classes": 80,
        "vram_min_mb": 1024,
        "expected_fps_t4": 70.0,
        "open_vocab": False,
        "rationale": "Kiến trúc mới nhất của Ultralytics, FPS cao hơn v8n ~5-10%.",
    },
    {
        "id": "OD-C",
        "name": "RT-DETR-r18 (Ultralytics)",
        "framework": "ultralytics",
        "model_id": "rtdetr-r18.pt",
        "model_size_mb_estimated": 20.4,
        "pretrained_classes": 80,
        "vram_min_mb": 1536,
        "expected_fps_t4": 45.0,
        "open_vocab": False,
        "rationale": "Real-time DETR, mAP cao hơn YOLO ở cùng FPS, transform-based.",
    },
    {
        "id": "OD-D",
        "name": "Florence-2-base (OD/Phrase Grounding)",
        "framework": "transformers",
        "model_id": "microsoft/Florence-2-base",
        "model_size_mb_estimated": 270.0,
        "pretrained_classes": -1,  # open-vocabulary
        "vram_min_mb": 4096,
        "expected_fps_t4": 5.0,
        "open_vocab": True,
        "rationale": "Đã có sẵn trong repo (vendored). Open-vocab, nhưng chậm hơn YOLO nhiều.",
    },
    {
        "id": "OD-E",
        "name": "GroundingDINO-base",
        "framework": "transformers",
        "model_id": "IDEA-Research/grounding-dino-base",
        "model_size_mb_estimated": 230.0,
        "pretrained_classes": -1,
        "vram_min_mb": 4096,
        "expected_fps_t4": 8.0,
        "open_vocab": True,
        "rationale": "Open-vocab mạnh, dùng text prompt. Đắt hơn YOLO, nhưng linh hoạt hơn.",
    },
]


@dataclass
class FrameSample:
    """Một frame đã trích xuất từ video, kèm bounding boxes ground truth (nếu có)."""
    video_id: str
    frame_idx: int
    image_path: str
    width: int
    height: int
    ground_truth_objects: List[Dict[str, Any]] = field(default_factory=list)
    # Ground truth format: [{"label": "person", "bbox_xyxy": [x1,y1,x2,y2]}, ...]


@dataclass
class ModelMeasurement:
    """Kết quả đo lường của 1 model trên 1 sample frame."""
    model_id: str
    video_id: str
    frame_idx: int
    num_detections: int
    avg_confidence: float
    inference_ms: float
    peak_vram_mb: float


@dataclass
class CandidateReport:
    """Báo cáo tổng hợp cho 1 candidate model."""
    id: str
    name: str
    rationale: str
    available: bool
    available_reason: str = ""
    model_size_mb_estimated: float = 0.0
    pretrained_classes: int = 0
    open_vocab: bool = False
    expected_fps_t4: float = 0.0
    measured_fps_t4: Optional[float] = None
    measured_mAP50: Optional[float] = None
    measured_precision: Optional[float] = None
    measured_recall: Optional[float] = None
    measured_peak_vram_mb: Optional[float] = None
    decision: str = "PENDING"  # PENDING / SELECTED / REJECTED
    decision_reason: str = ""


def detect_candidate_videos() -> List[Path]:
    """Chọn 3 video mẫu đa dạng domain để benchmark (lecture + business + data structures)."""
    if not CANDIDATE_VIDEOS_DIR.exists():
        return []
    candidates = []
    target_names = [
        "CA01.mp4",           # Education
        "Business Research_02 - Research Methods.mkv",  # Business
        "DS6-AlgorithmAnalysis.mp4",                    # CS / diagrams
    ]
    for name in target_names:
        p = CANDIDATE_VIDEOS_DIR / name
        if p.exists():
            candidates.append(p)
    # Nếu thiếu, fallback lấy 3 file đầu tiên
    if not candidates:
        for ext in ("*.mp4", "*.mkv", "*.webm"):
            for p in sorted(CANDIDATE_VIDEOS_DIR.glob(ext))[:3]:
                candidates.append(p)
    return candidates[:3]


def extract_sample_frames(video_path: Path, num_frames: int = 8) -> List[FrameSample]:
    """
    Trích frames cách đều từ video để benchmark.
    Returns empty list nếu OpenCV không khả dụng — caller xử lý fallback.
    """
    try:
        import cv2  # noqa
    except ImportError:
        return []

    frames: List[FrameSample] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // num_frames)
    out_dir = REPO_ROOT / "experiments" / "scratch" / "detection_frames" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while saved < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            break
        out_path = out_dir / f"frame_{saved:03d}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(FrameSample(
            video_id=video_path.stem,
            frame_idx=saved,
            image_path=str(out_path),
            width=width,
            height=height,
        ))
        frame_idx += step
        saved += 1
    cap.release()
    return frames


def check_availability() -> Dict[str, bool]:
    """Kiểm tra package nào đã cài đặt để quyết định candidate nào có thể chạy."""
    availability = {}
    for cand in DETECTION_CANDIDATES:
        if cand["framework"] == "ultralytics":
            try:
                import ultralytics  # noqa
                availability[cand["id"]] = True
            except ImportError:
                availability[cand["id"]] = False
        elif cand["framework"] == "transformers":
            try:
                import transformers  # noqa
                availability[cand["id"]] = True
            except ImportError:
                availability[cand["id"]] = False
        else:
            availability[cand["id"]] = False
    return availability


def measure_ultralytics_model(
    model_id: str,
    frames: List[FrameSample],
    imgsz: int = 640,
    conf_threshold: float = 0.25,
) -> List[ModelMeasurement]:
    """Chạy YOLO/RT-DETR và đo FPS + detection counts."""
    from ultralytics import YOLO  # type: ignore
    import torch

    measurements: List[ModelMeasurement] = []
    model = YOLO(model_id)
    # warmup
    if frames:
        _ = model(frames[0].image_path, verbose=False, imgsz=imgsz)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for fr in frames:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        results = model(fr.image_path, verbose=False, imgsz=imgsz, conf=conf_threshold)
        dt = time.time() - t0
        n = 0
        confs: List[float] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            n += len(boxes)
            confs.extend(float(c) for c in boxes.conf.cpu().numpy().tolist())
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available() else 0.0
        )
        measurements.append(ModelMeasurement(
            model_id=model_id,
            video_id=fr.video_id,
            frame_idx=fr.frame_idx,
            num_detections=n,
            avg_confidence=float(sum(confs) / len(confs)) if confs else 0.0,
            inference_ms=dt * 1000.0,
            peak_vram_mb=peak_mb,
        ))
    return measurements


def measure_florence2_model(
    frames: List[FrameSample], task: str = "<OD>"
) -> List[ModelMeasurement]:
    """
    Florence-2 open-vocabulary detection: chạy task <OD> với text prompt mặc định
    (các object phổ biến trong lecture/business/CS video).
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM  # type: ignore

    measurements: List[ModelMeasurement] = []
    vendored = REPO_ROOT / "ai_workers" / "modules" / "visual_v2" / "florence2_vendor"
    if (vendored / "model.safetensors").exists():
        proc = AutoProcessor.from_pretrained(str(vendored), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(vendored), trust_remote_code=True, torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    else:
        proc = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # warmup
    if frames:
        with torch.inference_mode():
            _ = model.generate(
                input_ids=proc.tokenizer("<OD>", return_tensors="pt").input_ids.to(
                    model.device
                ),
                max_new_tokens=64,
            )

    text_prompt = "person, laptop, book, screen, table, whiteboard, slide"
    for fr in frames:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        image = Image.open(fr.image_path).convert("RGB")
        t0 = time.time()
        with torch.inference_mode():
            inputs = proc(text=text_prompt, images=image, return_tensors="pt").to(
                model.device
            )
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].half()
                if model.device.type != "cpu" else inputs["pixel_values"],
                max_new_tokens=128,
            )
        gen_text = proc.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(
            gen_text, task=task, image_size=(image.width, image.height)
        )
        dt = time.time() - t0
        od_result = parsed.get(task, {})
        bboxes = od_result.get("bboxes", []) if isinstance(od_result, dict) else []
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available() else 0.0
        )
        measurements.append(ModelMeasurement(
            model_id="microsoft/Florence-2-base",
            video_id=fr.video_id,
            frame_idx=fr.frame_idx,
            num_detections=len(bboxes),
            avg_confidence=0.0,  # Florence-2 OD không trả confidence
            inference_ms=dt * 1000.0,
            peak_vram_mb=peak_mb,
        ))
    return measurements


def aggregate(measurements: List[ModelMeasurement]) -> Dict[str, float]:
    if not measurements:
        return {"fps": 0.0, "peak_vram_mb": 0.0, "avg_detections": 0.0, "avg_confidence": 0.0}
    fps = 1000.0 / (sum(m.inference_ms for m in measurements) / len(measurements))
    return {
        "fps": round(fps, 2),
        "peak_vram_mb": round(max(m.peak_vram_mb for m in measurements), 1),
        "avg_detections": round(sum(m.num_detections for m in measurements) / len(measurements), 2),
        "avg_confidence": round(sum(m.avg_confidence for m in measurements) / len(measurements), 4),
    }


def make_decision(report: CandidateReport) -> CandidateReport:
    """
    Heuristic quyết định dựa trên dữ liệu đo được:
    - Cân bằng FPS + VRAM + mAP coverage
    - Ưu tiên YOLO/RT-DETR nếu T4 đủ và detection count > 0
    - Florence-2 chỉ chọn nếu open-vocab quan trọng và budget VRAM cho phép
    """
    if not report.available:
        report.decision = "REJECTED"
        report.decision_reason = "Package chưa cài. Chạy `pip install ultralytics` để đánh giá."
        return report

    if report.measured_fps_t4 is None:
        report.decision = "PENDING"
        report.decision_reason = "Chưa đo trên T4. Cần chạy benchmark thực tế trên GPU."
        return report

    fps = report.measured_fps_t4
    vram = report.measured_peak_vram_mb or 0.0

    # Hard constraints: T4 16GB, mỗi phase phải tránh OOM
    if vram > 4096 and fps < 10:
        report.decision = "REJECTED"
        report.decision_reason = f"VRAM={vram}MB quá cao, FPS={fps} quá thấp cho real-time pipeline."
    elif fps < 8:
        report.decision = "REJECTED"
        report.decision_reason = f"FPS={fps} quá thấp — ảnh hưởng latency pipeline."
    elif report.open_vocab and vram > 2048:
        report.decision = "REJECTED"
        report.decision_reason = "Open-vocab Florence-2 tốn VRAM — không đáng cho lecture domain (COCO đủ)."
    elif not report.open_vocab and fps >= 30 and vram <= 2048:
        report.decision = "SELECTED"
        report.decision_reason = (
            f"Đáp ứng realtime (FPS={fps}), VRAM an toàn ({vram}MB), COCO 80 classes đủ dùng."
        )
    else:
        report.decision = "PENDING"
        report.decision_reason = "Cần so sánh thêm với các candidate khác."
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset candidate IDs, ví dụ: OD-A OD-B OD-D")
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidate-id", type=str, default=None,
                   help="Full registry candidate ID echoed in ablation output")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Unique result path for one ablation combination")
    p.add_argument("--run-id", type=str, default="standalone")
    p.add_argument("--config-hash", type=str, default="standalone")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ in candidate metadata, không chạy model (cho Colab/CI quick check)")
    args = p.parse_args(argv)

    if not 0 < args.conf_threshold <= 1:
        p.error("--conf-threshold must be in (0, 1]")
    random.seed(args.seed)

    print("=" * 70)
    print("RQ-OD: Object Detection Benchmark — Tìm model tối ưu cho lecture + multi-object")
    print("=" * 70)

    availability = check_availability()
    print("\n[Availability]")
    for cid, ok in availability.items():
        cand = next(c for c in DETECTION_CANDIDATES if c["id"] == cid)
        status = "✓" if ok else "✗"
        print(f"  {status} {cid:6s} {cand['name']:50s} (cần {cand['framework']})")

    # Lọc candidate theo flag --models
    selected_ids = set(args.models) if args.models else {c["id"] for c in DETECTION_CANDIDATES}
    videos = detect_candidate_videos()
    print(f"\n[Candidate Videos] {len(videos)} video mẫu:")
    for v in videos:
        print(f"  - {v.name}")

    all_frames: List[FrameSample] = []
    for v in videos:
        fr = extract_sample_frames(v, num_frames=args.frames_per_video)
        print(f"  → Trích {len(fr)} frames từ {v.name}")
        all_frames.extend(fr)
    print(f"\n[Frames] {len(all_frames)} tổng cộng.")

    reports: List[CandidateReport] = []
    for cand in DETECTION_CANDIDATES:
        if cand["id"] not in selected_ids:
            continue
        rep = CandidateReport(
            id=cand["id"],
            name=cand["name"],
            rationale=cand["rationale"],
            available=availability.get(cand["id"], False),
            model_size_mb_estimated=cand["model_size_mb_estimated"],
            pretrained_classes=cand["pretrained_classes"],
            open_vocab=cand["open_vocab"],
            expected_fps_t4=cand["expected_fps_t4"],
            available_reason=("OK" if availability.get(cand["id"], False)
                              else "Chưa cài package"),
        )
        if not rep.available or args.dry_run or not all_frames:
            reports.append(make_decision(rep))
            continue
        try:
            if cand["framework"] == "ultralytics":
                ms = measure_ultralytics_model(
                    cand["model_id"],
                    all_frames,
                    imgsz=args.imgsz,
                    conf_threshold=args.conf_threshold,
                )
            elif cand["id"] == "OD-D":
                ms = measure_florence2_model(all_frames)
            else:
                ms = []
            agg = aggregate(ms)
            rep.measured_fps_t4 = agg["fps"]
            rep.measured_peak_vram_mb = agg["peak_vram_mb"]
            rep.measured_precision = None  # cần ground truth để tính
            rep.measured_recall = None
            rep.measured_mAP50 = None
        except Exception as e:
            rep.available = True
            rep.decision_reason = f"Lỗi khi chạy: {e}"
        reports.append(make_decision(rep))

    # In bảng tóm tắt
    print("\n" + "=" * 70)
    print("TÓM TẮT CANDIDATES")
    print("=" * 70)
    print(f"{'ID':<6}{'Name':<35}{'Avail':<7}{'FPS':<8}{'VRAM(MB)':<11}{'Decision':<12}")
    print("-" * 70)
    for r in reports:
        fps = f"{r.measured_fps_t4:.1f}" if r.measured_fps_t4 else "n/a"
        vram = f"{r.measured_peak_vram_mb:.0f}" if r.measured_peak_vram_mb else "n/a"
        print(f"{r.id:<6}{r.name[:33]:<35}{('Y' if r.available else 'N'):<7}{fps:<8}{vram:<11}{r.decision:<12}")
        print(f"     → {r.decision_reason}")

    # Ghi JSON
    out_path = args.output_json or RESULTS_DIR / "rq_od_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    measured_metrics: Dict[str, float] = {}
    if len(reports) == 1:
        report = reports[0]
        for metric_name, value in (
            ("fps", report.measured_fps_t4),
            ("peak_vram_mb", report.measured_peak_vram_mb),
            ("precision", report.measured_precision),
            ("recall", report.measured_recall),
            ("map50", report.measured_mAP50),
        ):
            if value is not None:
                measured_metrics[metric_name] = float(value)
    payload = {
        "experiment": "RQ-OD_Object_Detection_Benchmark",
        "purpose": "Đánh giá nhiều model detection để chọn model tối ưu bổ sung cho đề cương (mục 2 + 3.1)",
        "candidates_evaluated": [asdict(r) for r in reports],
        "frame_samples": len(all_frames),
        "videos_used": [v.name for v in videos],
        "hardware_target": "Tesla T4 16GB / CPU fallback",
        "decision_rule": (
            "SELECTED: FPS>=30 AND VRAM<=2048MB AND not open_vocab "
            "(Florence-2 chỉ chọn nếu open-vocab quan trọng và VRAM cho phép)"
        ),
        "ablation_contract": {
            "task": "RQ0_OD",
            "candidate_id": args.candidate_id,
            "toggle_values": {
                "conf_threshold": args.conf_threshold,
                "imgsz": args.imgsz,
            },
            "seed": args.seed,
        } if args.candidate_id else None,
        "metrics": measured_metrics,
        "evidence_status": "diagnostic",
        "provenance": {
            "run_id": args.run_id,
            "config_hash": args.config_hash,
            "runner_version": "rq0_od_v1",
            "frame_count": len(all_frames),
            "note": "No released box/class ground truth is wired; diagnostic only.",
        },
    }
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(out_path)
    print(f"\n[Output] {out_path}")

    selected = [r for r in reports if r.decision == "SELECTED"]
    print(f"\n[Recommendation] {len(selected)} model SELECTED:")
    for r in selected:
        print(f"  ★ {r.id}: {r.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
