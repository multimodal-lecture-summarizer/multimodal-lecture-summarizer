/* Hallmark · component: VideoPreflightModal · genre: modern-minimal · theme: editorial-technical
 * states: idle · loading · scanning · step_passed · step_failed · all_passed · halted
 * microinteractions: tactile sequential scanner, monospace technical badges, auto-advance on pass
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  Film, 
  ArrowRight, 
  ShieldCheck, 
  Cpu, 
  Sparkles,
  ExternalLink
} from 'lucide-react';
import { api } from '../services/api';

import { 
  extractClientVideoMetadata, 
  buildClientPreflightData,
  normalizeVideoStandards,
  type VideoStandardConfig
} from '../utils/videoValidator';
import type { SummaryLength } from '../types';

export interface PreflightCheckItem {
  id: string;
  name: string;
  detected: string;
  standard: string;
  passed: boolean;
  message: string;
}

export interface PreflightData {
  overallValid: boolean;
  fileName: string;
  sizeMb: number;
  duration: number;
  resolution: string;
  videoCodec: string | null;
  audioCodec: string | null;
  fps: number;
  checks: PreflightCheckItem[];
}

interface VideoPreflightModalProps {
  isOpen: boolean;
  onClose: () => void;
  onProceed: () => void;
  file: File | null;
  url: string | null;
  preloadedData?: PreflightData | null;
  standards?: VideoStandardConfig | null;
  summaryLength?: SummaryLength;
  onSummaryLengthChange?: (length: SummaryLength) => void;
}

export const VideoPreflightModal: React.FC<VideoPreflightModalProps> = ({
  isOpen,
  onClose,
  onProceed,
  file,
  url,
  preloadedData,
  standards,
  summaryLength,
  onSummaryLengthChange,
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PreflightData | null>(null);

  // Sequential scanning state
  const [activeStepIndex, setActiveStepIndex] = useState<number>(-1);
  const [passedSteps, setPassedSteps] = useState<number[]>([]);
  const [failedStepIndex, setFailedStepIndex] = useState<number | null>(null);
  const [isAllCompleted, setIsAllCompleted] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  const scanTimeoutRef = useRef<any>(null);
  const countdownIntervalRef = useRef<any>(null);
  const hasProceededRef = useRef(false);

  const handleProceedSafe = () => {
    if (hasProceededRef.current) return;
    hasProceededRef.current = true;
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    setCountdown(null);
    onProceed();
  };

  // Fetch inspection data when modal opens
  useEffect(() => {
    if (!isOpen) {
      // Reset states
      hasProceededRef.current = false;
      setLoading(true);
      setError(null);
      setData(null);
      setActiveStepIndex(-1);
      setPassedSteps([]);
      setFailedStepIndex(null);
      setIsAllCompleted(false);
      setCountdown(null);
      if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
      return;
    }

    let isMounted = true;
    hasProceededRef.current = false;
    setLoading(true);
    setError(null);

    // Fast-path 1: Pre-computed client preflight data provided directly
    if (preloadedData) {
      setData(preloadedData);
      setLoading(false);
      scanTimeoutRef.current = setTimeout(() => {
        if (isMounted) setActiveStepIndex(0);
      }, 350);
      return;
    }

    // Fast-path 2: Local File - extract metadata 100% in browser (0ms network upload)
    if (file) {
      extractClientVideoMetadata(file)
        .then((meta) => {
          if (!isMounted) return;
          const std = standards || normalizeVideoStandards(null);
          const clientData = buildClientPreflightData(file, meta, std);
          setData(clientData);
          setLoading(false);
          scanTimeoutRef.current = setTimeout(() => {
            if (isMounted) setActiveStepIndex(0);
          }, 350);
        })
        .catch((err) => {
          if (!isMounted) return;
          setLoading(false);
          setError(err.message || "Không thể giải mã tệp video trên trình duyệt.");
        });
      return;
    }

    // Remote URL (YouTube): must inspect on server via yt-dlp
    api.inspectPreflight(null, url)
      .then((res) => {
        if (!isMounted) return;
        if (res.success && res.data) {
          setData(res.data);
          setLoading(false);
          // Start sequential scanner after brief layout settle
          scanTimeoutRef.current = setTimeout(() => {
            if (isMounted) setActiveStepIndex(0);
          }, 350);
        } else {
          throw new Error(res.message || "Không thể phân tích luồng video.");
        }
      })
      .catch((err: any) => {
        if (!isMounted) return;
        setLoading(false);
        setError(err.message || "Lỗi khi trích xuất thông số kỹ thuật của video.");
      });

    return () => {
      isMounted = false;
      if (scanTimeoutRef.current) clearTimeout(scanTimeoutRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    };
  }, [isOpen, file, url]);

  // Sequential cursor scanner progression
  useEffect(() => {
    if (!data || activeStepIndex < 0 || failedStepIndex !== null || isAllCompleted) {
      return;
    }

    const checks = data.checks;
    if (activeStepIndex >= checks.length) {
      // All checked!
      setIsAllCompleted(true);
      setCountdown(2);
      return;
    }

    const currentCheck = checks[activeStepIndex];

    // Hold on current step to show scanner cursor moving
    const timer = setTimeout(() => {
      if (currentCheck.passed) {
        setPassedSteps((prev) => [...prev, activeStepIndex]);
        setActiveStepIndex(activeStepIndex + 1);
      } else {
        // Failed at this step! Halt cursor here!
        setFailedStepIndex(activeStepIndex);
      }
    }, 420); // 420ms tactile cadence per row

    return () => clearTimeout(timer);
  }, [activeStepIndex, data, failedStepIndex, isAllCompleted]);

  // Countdown auto-proceed effect
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      handleProceedSafe();
      return;
    }

    countdownIntervalRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev !== null && prev <= 1) {
          return 0;
        }
        return prev !== null ? prev - 1 : 0;
      });
    }, 1000);

    return () => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
      }
    };
  }, [countdown]);

  if (!isOpen) return null;

  const totalChecks = data?.checks?.length || 7;
  const passedCount = passedSteps.length;
  const isFailed = failedStepIndex !== null;

  return (
    <AnimatePresence>
      <div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-sm"
        aria-modal="true"
        role="dialog"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 12 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="w-full max-w-lg bg-white rounded-2xl shadow-2xl border border-slate-200/90 overflow-hidden flex flex-col max-h-[90vh]"
        >
          {/* Header Bar */}
          <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
            <div className="flex items-center gap-3">
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                isAllCompleted 
                  ? 'bg-emerald-100 text-emerald-700' 
                  : isFailed 
                    ? 'bg-rose-100 text-rose-700'
                    : 'bg-primary/10 text-primary'
              }`}>
                {isAllCompleted ? (
                  <ShieldCheck className="w-5 h-5 text-emerald-600" />
                ) : isFailed ? (
                  <AlertTriangle className="w-5 h-5 text-rose-600" />
                ) : (
                  <Film className="w-4 h-4 text-primary animate-pulse" />
                )}
              </div>
              <div>
                <h3 className="font-heading text-sm md:text-base font-bold text-slate-900 leading-tight">
                  Thẩm định Tiêu chuẩn Video
                </h3>
                <p className="text-xs text-slate-500 font-mono truncate max-w-[280px]">
                  {file?.name || url || "Đang chuẩn bị tệp..."}
                </p>
              </div>
            </div>

            {/* Counter pill */}
            <div className="flex items-center gap-2">
              <span className={`text-[11px] font-mono px-2 py-0.5 rounded-full border ${
                isAllCompleted
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : isFailed
                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                    : 'bg-slate-100 text-slate-700 border-slate-200'
              }`}>
                {loading 
                  ? "ffprobe" 
                  : isAllCompleted 
                    ? "✓ ĐẠT CHUẨN" 
                    : isFailed 
                      ? "✕ TỪ CHỐI" 
                      : `${passedCount}/${totalChecks}`}
              </span>
            </div>
          </div>

          {/* Body Content */}
          <div className="p-5 overflow-y-auto flex-1 space-y-4">
            {/* Loading Skeleton */}
            {loading && (
              <div className="py-8 px-4 text-center space-y-3">
                <div className="inline-flex items-center justify-center p-3 rounded-full bg-slate-100 text-primary animate-spin">
                  <Cpu className="w-6 h-6" />
                </div>
                <p className="text-sm font-semibold text-slate-800">
                  Đang bóc tách luồng dữ liệu video qua ffprobe...
                </p>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Kiểm tra container, video/audio codec, resolution matrix và framerate.
                </p>
                <div className="w-48 h-1.5 bg-slate-100 rounded-full mx-auto overflow-hidden">
                  <div className="w-full h-full bg-primary/70 animate-pulse" />
                </div>
              </div>
            )}

            {/* Top Level Error */}
            {!loading && error && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs space-y-2">
                <div className="flex items-center gap-2 font-semibold">
                  <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
                  <span>Không thể trích xuất thông số tệp video</span>
                </div>
                <p className="text-rose-700 leading-relaxed font-mono">{error}</p>
              </div>
            )}

            {/* Checklist View */}
            {!loading && data && (
              <div className="space-y-2">
                <div className="text-[11px] uppercase tracking-wider font-bold text-slate-600 mb-1 flex items-center justify-between">
                  <span>Quy chuẩn kỹ thuật đầu vào</span>
                  <span>Kết quả kiểm định</span>
                </div>

                <div className="space-y-2 border border-slate-100 rounded-xl p-2 bg-slate-50/30">
                  {data.checks.map((check, index) => {
                    const isPassed = passedSteps.includes(index);
                    const isScanning = activeStepIndex === index && failedStepIndex === null && !isAllCompleted;
                    const isFailedRow = failedStepIndex === index;
                    const isPending = !isPassed && !isScanning && !isFailedRow;

                    return (
                      <motion.div
                        key={check.id}
                        initial={false}
                        animate={{
                          backgroundColor: isScanning 
                            ? 'rgba(248, 250, 252, 1)' 
                            : isFailedRow 
                              ? 'rgba(254, 242, 242, 0.8)' 
                              : isPassed 
                                ? 'rgba(255, 255, 255, 1)' 
                                : 'rgba(248, 250, 252, 0.4)',
                          borderColor: isScanning 
                            ? '#a855f7' 
                            : isFailedRow 
                              ? '#f43f5e' 
                              : isPassed 
                                ? '#e2e8f0' 
                                : '#f1f5f9'
                        }}
                        transition={{ duration: 0.15 }}
                        className={`relative rounded-lg p-2.5 border transition-all duration-150 flex flex-col gap-1 ${
                          isScanning ? 'shadow-sm ring-1 ring-primary/20' : ''
                        }`}
                      >
                        {/* Scanning Laser Cursor Indicator */}
                        {isScanning && (
                          <motion.div
                            layoutId="scanner-cursor"
                            className="absolute -left-1 top-2.5 bottom-2.5 w-1.5 bg-primary rounded-full shadow-[0_0_8px_rgba(168,85,247,0.8)]"
                            transition={{ type: 'spring', stiffness: 350, damping: 25 }}
                          />
                        )}

                        <div className="flex items-center justify-between gap-3">
                          {/* Left: Index & Name */}
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="font-mono text-[10px] text-slate-600 w-4">
                              0{index + 1}
                            </span>
                            <span className={`text-xs font-semibold truncate ${
                              isFailedRow ? 'text-rose-900 font-bold' : isScanning ? 'text-slate-900 font-bold' : 'text-slate-800'
                            }`}>
                              {check.name}
                            </span>
                          </div>

                          {/* Right: Badge & Status Icon */}
                          <div className="flex items-center gap-2 shrink-0">
                            {/* Detected Spec Monospace Pill */}
                            <span className={`font-mono text-[11px] px-2 py-0.5 rounded border ${
                              isPassed
                                ? 'bg-emerald-50 text-emerald-800 border-emerald-200/80'
                                : isFailedRow
                                  ? 'bg-rose-100 text-rose-800 border-rose-300 font-bold'
                                  : isScanning
                                    ? 'bg-primary/10 text-primary border-primary/20 animate-pulse'
                                    : 'bg-slate-100 text-slate-600 border-slate-200/60'
                            }`}>
                              {check.detected}
                            </span>

                            {/* Status Icon */}
                            <div className="w-5 h-5 flex items-center justify-center">
                              {isPassed && (
                                <motion.div
                                  initial={{ scale: 0.5, opacity: 0 }}
                                  animate={{ scale: 1, opacity: 1 }}
                                  transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                                >
                                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                                </motion.div>
                              )}
                              {isFailedRow && (
                                <motion.div
                                  initial={{ scale: 0.5, opacity: 0 }}
                                  animate={{ scale: 1, opacity: 1 }}
                                  transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                                >
                                  <XCircle className="w-4 h-4 text-rose-600" />
                                </motion.div>
                              )}
                              {isScanning && (
                                <div className="w-3.5 h-3.5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                              )}
                              {isPending && (
                                <div className="w-2 h-2 rounded-full bg-slate-300" />
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Detail / Error Explanation when Failed */}
                        {isFailedRow && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            className="mt-1 pt-1.5 border-t border-rose-200/60 text-[11px] text-rose-700 flex items-start gap-1.5"
                          >
                            <AlertTriangle className="w-3.5 h-3.5 text-rose-600 shrink-0 mt-0.5" />
                            <span className="leading-tight">{check.message}</span>
                          </motion.div>
                        )}
                      </motion.div>
                    );
                  })}
                </div>

                {/* All Passed Celebration Banner */}
                {isAllCompleted && (
                  <motion.div
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200/80 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2.5">
                      <Sparkles className="w-4 h-4 text-emerald-600" />
                      <div>
                        <p className="text-xs font-bold text-emerald-900">
                          Video đáp ứng toàn diện tiêu chuẩn hệ thống!
                        </p>
                        <p className="text-[11px] text-emerald-700">
                          {countdown !== null && countdown > 0 
                            ? `Tự động chuyển tiếp sau ${countdown}s...` 
                            : "Sẵn sàng đưa vào hàng đợi xử lý AI."}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Summary Depth Selection in Modal */}
                {isAllCompleted && summaryLength && (
                  <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-xl space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
                      <span>Cấp độ tóm tắt khởi tạo:</span>
                      <span className="text-[11px] font-medium text-primary font-bold">
                        {summaryLength === 'short' ? 'Ngắn gọn (TL;DR)' : summaryLength === 'detailed' ? 'Chi tiết (Chuyên sâu)' : 'Tiêu chuẩn (Tổng quan)'}
                      </span>
                    </div>
                    {onSummaryLengthChange && (
                      <div className="grid grid-cols-3 gap-1.5 p-1 bg-white border border-slate-200 rounded-lg text-[11px]">
                        <button
                          type="button"
                          onClick={() => {
                            if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
                            setCountdown(null);
                            onSummaryLengthChange('short');
                          }}
                          className={`py-1.5 px-2 rounded-md font-medium transition-all text-center ${
                            summaryLength === 'short'
                              ? 'bg-primary text-white font-bold shadow-xs'
                              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                          }`}
                        >
                          Ngắn gọn
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
                            setCountdown(null);
                            onSummaryLengthChange('medium');
                          }}
                          className={`py-1.5 px-2 rounded-md font-medium transition-all text-center ${
                            summaryLength === 'medium'
                              ? 'bg-primary text-white font-bold shadow-xs'
                              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                          }`}
                        >
                          Tiêu chuẩn
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
                            setCountdown(null);
                            onSummaryLengthChange('detailed');
                          }}
                          className={`py-1.5 px-2 rounded-md font-medium transition-all text-center ${
                            summaryLength === 'detailed'
                              ? 'bg-primary text-white font-bold shadow-xs'
                              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                          }`}
                        >
                          Chi tiết
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Footer Actions */}
          <div className="px-5 py-3.5 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-800 hover:bg-slate-200/60 rounded-lg transition-colors"
            >
              {isFailed ? "Hủy & Chọn tệp khác" : "Đóng"}
            </button>

            {isAllCompleted ? (
              <button
                type="button"
                onClick={handleProceedSafe}
                disabled={hasProceededRef.current}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-bold text-white bg-primary hover:bg-primary-hover active:scale-[0.98] rounded-lg shadow-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span>Bắt đầu tóm tắt ngay</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            ) : isFailed ? (
              <a
                href="#/admin"
                className="inline-flex items-center gap-1 text-[11px] text-slate-500 hover:text-primary underline"
              >
                <span>Xem cấu hình tiêu chuẩn</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            ) : null}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
