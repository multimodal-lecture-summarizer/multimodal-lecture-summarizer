/* Hallmark · component: PerformanceCorrelationWidget · genre: modern-minimal · theme: editorial-technical
 * states: default · hover · focus-visible · active · disabled · loading · error · success
 * microinteractions: tactile hover elevation, high-contrast monospace technical badges, interactive scatter plot tooltips
 * pre-emit critique: P5 H5 E5 S5 R5 V5
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import Chart from 'chart.js/auto';
import { api } from '../services/api';
import type { PerformanceAnalyticsData, ScatterDataPoint } from '../types';
import { 
  Zap, 
  Clock, 
  Film, 
  Activity, 
  AlertCircle, 
  RefreshCw, 
  Layers
} from 'lucide-react';

interface PerformanceCorrelationWidgetProps {
  onRefreshTrigger?: () => void;
}

export const PerformanceCorrelationWidget: React.FC<PerformanceCorrelationWidgetProps> = () => {
  const [data, setData] = useState<PerformanceAnalyticsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedBucketFilter, setSelectedBucketFilter] = useState<'all' | 'short' | 'medium' | 'long'>('all');

  const chartCanvasRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  const fetchPerformanceData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getPerformanceStats();
      if (res.success && res.data) {
        setData(res.data);
      } else {
        throw new Error(res.message || "Không thể tải dữ liệu phân tích hiệu năng.");
      }
    } catch (err: any) {
      console.error("Failed to load performance analytics:", err);
      setError(err.message || "Đã xảy ra lỗi khi kết nối tới máy chủ.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  // Filter scatter data points based on selected bucket
  const filteredScatterPoints = useMemo(() => {
    if (!data?.scatterData) return [];
    if (selectedBucketFilter === 'all') return data.scatterData;
    if (selectedBucketFilter === 'short') {
      return data.scatterData.filter((p) => p.durationSec < 300);
    }
    if (selectedBucketFilter === 'medium') {
      return data.scatterData.filter((p) => p.durationSec >= 300 && p.durationSec <= 900);
    }
    if (selectedBucketFilter === 'long') {
      return data.scatterData.filter((p) => p.durationSec > 900);
    }
    return data.scatterData;
  }, [data, selectedBucketFilter]);

  // Render & update Chart.js Scatter Plot
  useEffect(() => {
    if (!chartCanvasRef.current || !filteredScatterPoints.length || loading) {
      return;
    }

    // Destroy existing instance before creating a new one
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const ctx = chartCanvasRef.current.getContext('2d');
    if (!ctx) return;

    // Transform points: X in minutes (duration), Y in seconds (processing latency)
    const points = filteredScatterPoints.map((pt) => ({
      x: Number((pt.durationSec / 60).toFixed(2)),
      y: Number(pt.processingSec.toFixed(1)),
      raw: pt,
    }));

    chartInstanceRef.current = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Video đã xử lý',
            data: points,
            backgroundColor: points.map((p) => {
              if (p.raw.rtf < 0.25) return 'rgba(16, 185, 129, 0.85)'; // Emerald
              if (p.raw.rtf <= 0.45) return 'rgba(124, 58, 237, 0.85)'; // Primary Purple
              return 'rgba(245, 158, 11, 0.85)'; // Amber
            }),
            borderColor: points.map((p) => {
              if (p.raw.rtf < 0.25) return '#059669';
              if (p.raw.rtf <= 0.45) return '#6d28d9';
              return '#d97706';
            }),
            borderWidth: 1.5,
            pointRadius: 6,
            pointHoverRadius: 9,
            pointHoverBorderWidth: 2.5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 600,
          easing: 'easeOutQuart',
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#f8fafc',
            bodyColor: '#cbd5e1',
            padding: 12,
            boxPadding: 6,
            cornerRadius: 10,
            titleFont: {
              size: 13,
              weight: 'bold',
              family: 'Outfit, sans-serif',
            },
            bodyFont: {
              size: 11,
              family: 'JetBrains Mono, monospace',
            },
            callbacks: {
              title: (items) => {
                const raw = (items[0].raw as any)?.raw as ScatterDataPoint;
                return raw ? raw.title : 'Thông số Video';
              },
              label: (context) => {
                const raw = (context.raw as any)?.raw as ScatterDataPoint;
                if (!raw) return '';
                const durMin = Math.floor(raw.durationSec / 60);
                const durSec = Math.round(raw.durationSec % 60);
                return [
                  ` Độ dài video: ${durMin}m ${durSec}s (${raw.durationSec}s)`,
                  ` Thời gian xử lý: ${raw.processingSec}s`,
                  ` Hệ số RTF: ${raw.rtf.toFixed(2)}x (Nhanh gấp ${(1 / raw.rtf).toFixed(1)}x real-time)`,
                  ` Độ phân giải: ${raw.resolution}`,
                ];
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Độ dài video gốc (phút)',
              color: '#64748b',
              font: {
                size: 11,
                weight: 'bold',
                family: 'Outfit, sans-serif',
              },
            },
            grid: {
              color: 'rgba(226, 232, 240, 0.6)',
            },
            ticks: {
              color: '#64748b',
              font: {
                size: 10,
                family: 'JetBrains Mono, monospace',
              },
              callback: (val) => `${val}p`,
            },
          },
          y: {
            title: {
              display: true,
              text: 'Thời gian AI xử lý (giây)',
              color: '#64748b',
              font: {
                size: 11,
                weight: 'bold',
                family: 'Outfit, sans-serif',
              },
            },
            grid: {
              color: 'rgba(226, 232, 240, 0.6)',
            },
            ticks: {
              color: '#64748b',
              font: {
                size: 10,
                family: 'JetBrains Mono, monospace',
              },
              callback: (val) => `${val}s`,
            },
          },
        },
      },
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [filteredScatterPoints, loading]);

  const formatSecs = (sec: number): string => {
    const mins = Math.floor(sec / 60);
    const remainder = Math.round(sec % 60);
    if (mins === 0) return `${remainder}s`;
    return `${mins}m ${remainder}s`;
  };

  const getRtfBadgeClass = (rtf: number): string => {
    if (rtf < 0.5) {
      return 'bg-emerald-50 text-emerald-700 border-emerald-200/80';
    }
    if (rtf <= 1.0) {
      return 'bg-amber-50 text-amber-700 border-amber-200/80';
    }
    return 'bg-rose-50 text-rose-700 border-rose-200/80';
  };

  const getRtfDescription = (rtf: number): string => {
    if (rtf < 0.5) return 'Tối ưu (Nhanh vượt trội)';
    if (rtf <= 1.0) return 'Đạt chuẩn thời gian thực';
    return 'Cần tối ưu tài nguyên';
  };

  return (
    <div className="bg-white border-2 border-slate-200 rounded-2xl p-6 shadow-none space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-primary/10 text-primary">
              <Activity className="w-5 h-5" />
            </span>
            <h3 className="font-heading text-base font-bold text-slate-900 tracking-tight">
              Tương Quan Độ Dài Video & Thời Gian Xử Lý
            </h3>
          </div>
          <p className="text-xs text-slate-500 font-body">
            Đánh giá hiệu suất đa phương thức qua hệ số thời gian thực RTF (Thời gian xử lý / Thời lượng gốc).
          </p>
        </div>

        <button
          type="button"
          onClick={fetchPerformanceData}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors disabled:opacity-50 self-start sm:self-auto"
          title="Tải lại số liệu hiệu năng"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Làm mới</span>
        </button>
      </div>

      {/* Loading State */}
      {loading && !data && (
        <div className="space-y-4 py-8">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="h-24 bg-slate-100 animate-pulse rounded-xl" />
            <div className="h-24 bg-slate-100 animate-pulse rounded-xl" />
            <div className="h-24 bg-slate-100 animate-pulse rounded-xl" />
          </div>
          <div className="h-64 bg-slate-100 animate-pulse rounded-xl" />
          <div className="h-40 bg-slate-100 animate-pulse rounded-xl" />
        </div>
      )}

      {/* Error State */}
      {!loading && error && (
        <div className="p-6 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex flex-col items-center justify-center text-center gap-3">
          <AlertCircle className="w-8 h-8 text-rose-500" />
          <div>
            <h4 className="font-bold text-rose-900 text-sm">Không thể tải số liệu RTF</h4>
            <p className="text-rose-700 mt-1 max-w-md">{error}</p>
          </div>
          <button
            type="button"
            onClick={fetchPerformanceData}
            className="px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white rounded-lg font-bold text-xs shadow-sm transition-all"
          >
            Thử tải lại
          </button>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && data && data.scatterData.length === 0 && (
        <div className="py-12 text-center text-slate-500 space-y-2">
          <Film className="w-10 h-10 mx-auto text-slate-300" />
          <h4 className="font-bold text-slate-800 text-sm">Chưa có tác vụ hoàn thành</h4>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            Hệ thống cần ít nhất một video tóm tắt thành công để tổng hợp ma trận tương quan độ dài và thời gian xử lý.
          </p>
        </div>
      )}

      {/* Data State */}
      {!loading && !error && data && (
        <div className="space-y-6">
          {/* 1. KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Average RTF Card */}
            <div className="bg-slate-50/70 border border-slate-200/80 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden group hover:border-primary/40 transition-all">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Average RTF
                </span>
                <span className="p-1 rounded-md bg-primary/10 text-primary">
                  <Zap className="w-4 h-4" />
                </span>
              </div>
              <div>
                <div className="text-3xl font-black text-slate-900 font-mono tracking-tight flex items-baseline gap-1.5">
                  <span>{data.summaryMetrics.avgRtf.toFixed(2)}x</span>
                  <span className="text-xs font-normal text-emerald-600 font-sans font-bold">
                    (Nhanh gấp {(1 / (data.summaryMetrics.avgRtf || 1)).toFixed(1)}x)
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 mt-1 font-body">
                  Hệ số thời gian thực: chỉ mất {Math.round(data.summaryMetrics.avgRtf * 60)}s để xử lý 1 phút video.
                </p>
              </div>
            </div>

            {/* Average Processing Time Card */}
            <div className="bg-slate-50/70 border border-slate-200/80 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden group hover:border-primary/40 transition-all">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Thời gian Xử lý TB
                </span>
                <span className="p-1 rounded-md bg-indigo-50 text-indigo-600">
                  <Clock className="w-4 h-4" />
                </span>
              </div>
              <div>
                <div className="text-3xl font-black text-slate-900 font-mono tracking-tight">
                  {formatSecs(data.summaryMetrics.avgProcessingTimeSec)}
                </div>
                <p className="text-[11px] text-slate-500 mt-1 font-body">
                  Trung bình {data.summaryMetrics.avgProcessingTimeSec.toFixed(1)} giây trên toàn bộ tác vụ.
                </p>
              </div>
            </div>

            {/* Total Processed Hours Card */}
            <div className="bg-slate-50/70 border border-slate-200/80 rounded-xl p-4 flex flex-col justify-between relative overflow-hidden group hover:border-primary/40 transition-all">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Tổng thời lượng đã xử lý
                </span>
                <span className="p-1 rounded-md bg-emerald-50 text-emerald-600">
                  <Film className="w-4 h-4" />
                </span>
              </div>
              <div>
                <div className="text-3xl font-black text-slate-900 font-mono tracking-tight">
                  {data.summaryMetrics.totalProcessedHours.toFixed(1)} <span className="text-sm font-sans font-normal text-slate-500">giờ</span>
                </div>
                <p className="text-[11px] text-slate-500 mt-1 font-body">
                  Tổng hợp từ {data.summaryMetrics.totalJobsEvaluated} tác vụ pipeline AI hoàn thành.
                </p>
              </div>
            </div>
          </div>

          {/* 2. Scatter Plot: Correlation of Duration vs Processing Time */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h4 className="font-heading text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Biểu đồ phân tán: Độ dài Video (phút) vs Thời gian Xử lý (giây)
                </h4>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Mỗi điểm biểu diễn một bài giảng. Hover để xem tên video, độ phân giải và hệ số RTF chi tiết.
                </p>
              </div>

              {/* Bucket Filter Pills */}
              <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg self-start sm:self-auto text-[11px] font-semibold">
                <button
                  type="button"
                  onClick={() => setSelectedBucketFilter('all')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    selectedBucketFilter === 'all'
                      ? 'bg-white text-slate-900 shadow-sm font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Tất cả ({data.scatterData.length})
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedBucketFilter('short')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    selectedBucketFilter === 'short'
                      ? 'bg-white text-slate-900 shadow-sm font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  &lt; 5p
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedBucketFilter('medium')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    selectedBucketFilter === 'medium'
                      ? 'bg-white text-slate-900 shadow-sm font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  5 - 15p
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedBucketFilter('long')}
                  className={`px-2.5 py-1 rounded-md transition-all ${
                    selectedBucketFilter === 'long'
                      ? 'bg-white text-slate-900 shadow-sm font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  &gt; 15p
                </button>
              </div>
            </div>

            {/* Chart Canvas */}
            <div className="h-72 w-full relative">
              <canvas ref={chartCanvasRef} />
            </div>

            {/* Color Legend */}
            <div className="flex flex-wrap items-center justify-end gap-4 pt-2 border-t border-slate-100 text-[11px] text-slate-600">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span>RTF &lt; 0.25x (Rất nhanh)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-primary" />
                <span>RTF 0.25x - 0.45x (Tiêu chuẩn)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                <span>RTF &gt; 0.45x (Video dung lượng lớn)</span>
              </div>
            </div>
          </div>

          {/* 3. Duration Buckets Table */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="p-1 rounded bg-slate-100 text-slate-600">
                  <Layers className="w-4 h-4" />
                </span>
                <h4 className="font-heading text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Phân tích Phân khúc Thời lượng (Duration Buckets)
                </h4>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">
                3 phân khúc chuẩn hóa
              </span>
            </div>

            <div className="overflow-x-auto border border-slate-200 rounded-xl">
              <table className="w-full text-left text-xs border-collapse font-body">
                <thead>
                  <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-600 font-bold uppercase text-[10px] tracking-wider">
                    <th className="py-3 px-4">Phân khúc độ dài</th>
                    <th className="py-3 px-4 text-center">Số lượng video</th>
                    <th className="py-3 px-4">Thời lượng TB</th>
                    <th className="py-3 px-4">Thời gian xử lý TB</th>
                    <th className="py-3 px-4">Hệ số RTF TB</th>
                    <th className="py-3 px-4 text-right">Tỷ lệ thành công</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-800">
                  {data.durationBuckets.map((bucket) => {
                    const badgeClass = getRtfBadgeClass(bucket.avgRtf);
                    const rtfDesc = getRtfDescription(bucket.avgRtf);

                    return (
                      <tr 
                        key={bucket.bucket} 
                        className="hover:bg-slate-50/60 transition-colors group"
                      >
                        <td className="py-3.5 px-4 font-semibold flex items-center gap-2 text-slate-900">
                          <span className={`w-2 h-2 rounded-full ${
                            bucket.bucket === 'short' 
                              ? 'bg-blue-500' 
                              : bucket.bucket === 'medium' 
                              ? 'bg-primary' 
                              : 'bg-purple-600'
                          }`} />
                          <div className="flex flex-col">
                            <span className="font-bold text-xs">{bucket.label}</span>
                            <span className="text-[10px] text-slate-400 font-mono font-normal">
                              {bucket.bucket === 'short' 
                                ? '< 300s' 
                                : bucket.bucket === 'medium' 
                                ? '300s - 900s' 
                                : '> 900s'}
                            </span>
                          </div>
                        </td>

                        <td className="py-3.5 px-4 text-center font-mono font-bold text-slate-800">
                          {bucket.count}
                        </td>

                        <td className="py-3.5 px-4 font-mono">
                          {formatSecs(bucket.avgDurationSec)}
                        </td>

                        <td className="py-3.5 px-4 font-mono font-semibold text-slate-900">
                          {formatSecs(bucket.avgProcessingSec)}
                        </td>

                        <td className="py-3.5 px-4">
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-0.5 rounded-full border text-[11px] font-mono font-bold ${badgeClass}`}>
                              {bucket.avgRtf.toFixed(2)}x
                            </span>
                            <span className="text-[10px] text-slate-500 hidden md:inline">
                              {rtfDesc}
                            </span>
                          </div>
                        </td>

                        <td className="py-3.5 px-4 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <div className="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden hidden sm:block">
                              <div
                                className="bg-emerald-500 h-1.5 rounded-full"
                                style={{ width: `${Math.min(100, bucket.successRate)}%` }}
                              />
                            </div>
                            <span className="font-mono font-bold text-emerald-600">
                              {bucket.successRate.toFixed(1)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
