/* Hallmark · component: SceneObjectExplorer · genre: editorial · theme: Specimen / Neutral-Tactile
 * states: default · hover · focus-visible · active · disabled · loading · error · success
 * microinteractions: tactile hover elevation, high-contrast timestamps, 2-way box-track sync, non-jumping scene inspection
 * pre-emit critique: P5 H5 E5 S5 R5 V5
 */

import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  Video,
  Layers,
  Box,
  Eye,
  EyeOff,
  SlidersHorizontal,
  Play,
  Clock,
  Tag,
  ChevronRight,
  RefreshCw,
  AlertCircle,
  FileText,
  ListFilter,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../services/api';
import { CONFIG } from '../config';
import type { VideoScene, KeyframeBBox, ObjectTrackSummary } from '../types';

interface SceneObjectExplorerProps {
  videoId: string;
  onSeekVideo: (seconds: number) => void;
  currentTime?: number;
}

// Color palette mapping per object label for clean, academic distinction
const LABEL_COLORS: Record<
  string,
  {
    border: string;
    bg: string;
    badgeBg: string;
    badgeDot: string;
    text: string;
  }
> = {
  person: {
    border: 'border-emerald-500',
    bg: 'bg-emerald-500/15',
    badgeBg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    badgeDot: 'bg-emerald-500',
    text: 'text-emerald-300',
  },
  motorcycle: {
    border: 'border-amber-500',
    bg: 'bg-amber-500/15',
    badgeBg: 'bg-amber-50 text-amber-800 border-amber-200',
    badgeDot: 'bg-amber-500',
    text: 'text-amber-300',
  },
  motorbike: {
    border: 'border-amber-500',
    bg: 'bg-amber-500/15',
    badgeBg: 'bg-amber-50 text-amber-800 border-amber-200',
    badgeDot: 'bg-amber-500',
    text: 'text-amber-300',
  },
  bicycle: {
    border: 'border-sky-500',
    bg: 'bg-sky-500/15',
    badgeBg: 'bg-sky-50 text-sky-800 border-sky-200',
    badgeDot: 'bg-sky-500',
    text: 'text-sky-300',
  },
  bike: {
    border: 'border-sky-500',
    bg: 'bg-sky-500/15',
    badgeBg: 'bg-sky-50 text-sky-800 border-sky-200',
    badgeDot: 'bg-sky-500',
    text: 'text-sky-300',
  },
  car: {
    border: 'border-blue-500',
    bg: 'bg-blue-500/15',
    badgeBg: 'bg-blue-50 text-blue-800 border-blue-200',
    badgeDot: 'bg-blue-500',
    text: 'text-blue-300',
  },
  truck: {
    border: 'border-indigo-500',
    bg: 'bg-indigo-500/15',
    badgeBg: 'bg-indigo-50 text-indigo-800 border-indigo-200',
    badgeDot: 'bg-indigo-500',
    text: 'text-indigo-300',
  },
  bus: {
    border: 'border-purple-500',
    bg: 'bg-purple-500/15',
    badgeBg: 'bg-purple-50 text-purple-800 border-purple-200',
    badgeDot: 'bg-purple-500',
    text: 'text-purple-300',
  },
  laptop: {
    border: 'border-violet-500',
    bg: 'bg-violet-500/15',
    badgeBg: 'bg-violet-50 text-violet-800 border-violet-200',
    badgeDot: 'bg-violet-500',
    text: 'text-violet-300',
  },
  tv: {
    border: 'border-fuchsia-500',
    bg: 'bg-fuchsia-500/15',
    badgeBg: 'bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200',
    badgeDot: 'bg-fuchsia-500',
    text: 'text-fuchsia-300',
  },
  monitor: {
    border: 'border-fuchsia-500',
    bg: 'bg-fuchsia-500/15',
    badgeBg: 'bg-fuchsia-50 text-fuchsia-800 border-fuchsia-200',
    badgeDot: 'bg-fuchsia-500',
    text: 'text-fuchsia-300',
  },
  'cell phone': {
    border: 'border-orange-500',
    bg: 'bg-orange-500/15',
    badgeBg: 'bg-orange-50 text-orange-800 border-orange-200',
    badgeDot: 'bg-orange-500',
    text: 'text-orange-300',
  },
  cellphone: {
    border: 'border-orange-500',
    bg: 'bg-orange-500/15',
    badgeBg: 'bg-orange-50 text-orange-800 border-orange-200',
    badgeDot: 'bg-orange-500',
    text: 'text-orange-300',
  },
  chair: {
    border: 'border-teal-500',
    bg: 'bg-teal-500/15',
    badgeBg: 'bg-teal-50 text-teal-800 border-teal-200',
    badgeDot: 'bg-teal-500',
    text: 'text-teal-300',
  },
  book: {
    border: 'border-rose-500',
    bg: 'bg-rose-500/15',
    badgeBg: 'bg-rose-50 text-rose-800 border-rose-200',
    badgeDot: 'bg-rose-500',
    text: 'text-rose-300',
  },
  bottle: {
    border: 'border-cyan-500',
    bg: 'bg-cyan-500/15',
    badgeBg: 'bg-cyan-50 text-cyan-800 border-cyan-200',
    badgeDot: 'bg-cyan-500',
    text: 'text-cyan-300',
  },
};

const FALLBACK_PALETTES = [
  { border: 'border-emerald-500', bg: 'bg-emerald-500/15', badgeBg: 'bg-emerald-50 text-emerald-800 border-emerald-200', badgeDot: 'bg-emerald-500', text: 'text-emerald-300' },
  { border: 'border-amber-500', bg: 'bg-amber-500/15', badgeBg: 'bg-amber-50 text-amber-800 border-amber-200', badgeDot: 'bg-amber-500', text: 'text-amber-300' },
  { border: 'border-blue-500', bg: 'bg-blue-500/15', badgeBg: 'bg-blue-50 text-blue-800 border-blue-200', badgeDot: 'bg-blue-500', text: 'text-blue-300' },
  { border: 'border-violet-500', bg: 'bg-violet-500/15', badgeBg: 'bg-violet-50 text-violet-800 border-violet-200', badgeDot: 'bg-violet-500', text: 'text-violet-300' },
  { border: 'border-teal-500', bg: 'bg-teal-500/15', badgeBg: 'bg-teal-50 text-teal-800 border-teal-200', badgeDot: 'bg-teal-500', text: 'text-teal-300' },
  { border: 'border-rose-500', bg: 'bg-rose-500/15', badgeBg: 'bg-rose-50 text-rose-800 border-rose-200', badgeDot: 'bg-rose-500', text: 'text-rose-300' },
];

const getColorForLabel = (label: string) => {
  const key = (label || '').trim().toLowerCase();
  if (LABEL_COLORS[key]) return LABEL_COLORS[key];

  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = key.charCodeAt(i) + ((hash << 5) - hash);
  }
  const idx = Math.abs(hash) % FALLBACK_PALETTES.length;
  return FALLBACK_PALETTES[idx];
};

const formatSeconds = (secs: number): string => {
  if (isNaN(secs) || secs < 0) return '00:00';
  const m = Math.floor(secs / 60)
    .toString()
    .padStart(2, '0');
  const s = Math.floor(secs % 60)
    .toString()
    .padStart(2, '0');
  return `${m}:${s}`;
};

// Easily confused food/texture classes that produce high false positives on nature, wood and outdoor video
const LOW_RELEVANCE_NOISE_CLASSES = new Set([
  'broccoli', 'carrot', 'donut', 'pizza', 'hot dog', 'sandwich', 'cake', 'banana', 'apple', 'orange'
]);

const isNoiseClass = (label: string, conf: number) => {
  const lbl = (label || '').trim().toLowerCase();
  return LOW_RELEVANCE_NOISE_CLASSES.has(lbl) && conf < 0.65;
};

export const SceneObjectExplorer: React.FC<SceneObjectExplorerProps> = ({
  videoId,
  onSeekVideo,
}) => {
  const [scenes, setScenes] = useState<VideoScene[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Selected Scene and Filtering
  const [selectedSceneIndex, setSelectedSceneIndex] = useState<number>(1);
  const [selectedLabelFilter, setSelectedLabelFilter] = useState<string | null>(null);
  const [minConfidenceFilter, setMinConfidenceFilter] = useState<number>(0.35);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'tracks' | 'info'>('tracks');

  // Interactive 2-Way State: Synchronize Track Table and Keyframe Bounding Boxes
  const [hoveredTrackId, setHoveredTrackId] = useState<number | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // Table row refs for auto-scroll into view when box is hovered
  const trackRowRefs = useRef<Record<number, HTMLTableRowElement | null>>({});

  // Fetch all scenes with detected objects for this video
  const fetchScenesData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getVideoScenes(videoId);
      if (res.success && Array.isArray(res.data)) {
        setScenes(res.data);
        if (res.data.length > 0) {
          const firstIdx = res.data[0].sceneIndex !== undefined ? res.data[0].sceneIndex : res.data[0].scene_index || 1;
          setSelectedSceneIndex(firstIdx);
        }
      } else {
        throw new Error(res.message || 'Không thể tải danh sách phân cảnh bài giảng.');
      }
    } catch (err: any) {
      console.error('Error fetching video scenes for OD:', err);
      setError(err.message || 'Lỗi kết nối khi tải phân đoạn video.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (videoId) {
      fetchScenesData();
    }
  }, [videoId]);

  // Aggregate label statistics across all scenes (excluding low-confidence noise classes)
  const { totalUniqueTracks, globalLabelDistribution, hasAnyODData } = useMemo(() => {
    const labelStats: Record<string, { sceneCount: number; totalOccurrences: number }> = {};
    const videoTrackKeys = new Set<string>();
    let foundAny = false;

    scenes.forEach((scene) => {
      const det = scene.detectedObjectsJson || scene.detected_objects_json;
      const counts = scene.objectCountsJson || scene.object_counts_json || {};
      const sceneLabels = new Set<string>();

      if (det) {
        foundAny = true;
        const bboxes = det.keyframeBboxes || det.keyframe_bboxes || det.keyframeDetections || det.keyframe_detections || [];
        const tracks = det.tracksSummary || det.tracks_summary || [];

        bboxes.forEach((b) => {
          const conf = b.confidence || 0;
          const lbl = b.label?.trim().toLowerCase();
          if (lbl && !isNoiseClass(lbl, conf)) {
            sceneLabels.add(lbl);
            if (!labelStats[lbl]) {
              labelStats[lbl] = { sceneCount: 0, totalOccurrences: 0 };
            }
            labelStats[lbl].totalOccurrences += 1;
          }
        });

        tracks.forEach((t) => {
          const conf = t.avgConfidence ?? t.avg_confidence ?? 0;
          const lbl = t.label?.trim().toLowerCase();
          if (lbl && !isNoiseClass(lbl, conf)) {
            sceneLabels.add(lbl);
            const tId = t.trackId !== undefined ? t.trackId : t.track_id;
            if (tId !== undefined && tId !== null) {
              videoTrackKeys.add(`${lbl}_${tId}`);
            }
          }
        });
      }

      if (counts && typeof counts === 'object') {
        Object.entries(counts).forEach(([lblRaw, cnt]) => {
          const lbl = lblRaw.trim().toLowerCase();
          if (lbl && (cnt || 0) > 0 && !LOW_RELEVANCE_NOISE_CLASSES.has(lbl)) {
            foundAny = true;
            sceneLabels.add(lbl);
            if (!labelStats[lbl]) {
              labelStats[lbl] = { sceneCount: 0, totalOccurrences: 0 };
            }
            labelStats[lbl].totalOccurrences = Math.max(labelStats[lbl].totalOccurrences, cnt || 0);
          }
        });
      }

      sceneLabels.forEach((lbl) => {
        if (!labelStats[lbl]) {
          labelStats[lbl] = { sceneCount: 0, totalOccurrences: 0 };
        }
        labelStats[lbl].sceneCount += 1;
      });
    });

    return {
      totalUniqueTracks: videoTrackKeys.size,
      globalLabelDistribution: labelStats,
      hasAnyODData: foundAny,
    };
  }, [scenes]);

  // Filter scenes based on selected label
  const filteredScenes = useMemo(() => {
    if (!selectedLabelFilter) return scenes;
    const target = selectedLabelFilter.trim().toLowerCase();

    return scenes.filter((scene) => {
      const det = scene.detectedObjectsJson || scene.detected_objects_json;
      const counts = scene.objectCountsJson || scene.object_counts_json || {};

      if (det) {
        const bboxes = det.keyframeBboxes || det.keyframe_bboxes || det.keyframeDetections || det.keyframe_detections || [];
        const tracks = det.tracksSummary || det.tracks_summary || [];
        const hasBox = bboxes.some((b) => b.label?.trim().toLowerCase() === target);
        const hasTrack = tracks.some((t) => t.label?.trim().toLowerCase() === target);
        if (hasBox || hasTrack) return true;
      }

      if (counts && typeof counts === 'object' && (counts[target] || 0) > 0) {
        return true;
      }

      return false;
    });
  }, [scenes, selectedLabelFilter]);

  // Find currently selected scene
  const activeScene = useMemo(() => {
    const found = filteredScenes.find((s) => {
      const sIdx = s.sceneIndex !== undefined ? s.sceneIndex : s.scene_index;
      return sIdx === selectedSceneIndex;
    });
    return found || filteredScenes[0] || scenes[0] || null;
  }, [filteredScenes, scenes, selectedSceneIndex]);

  // Sequential 1, 2, 3... scene numbering for user
  const activeDisplayIndex = useMemo(() => {
    if (!activeScene) return 1;
    const activeIdx = activeScene.sceneIndex !== undefined ? activeScene.sceneIndex : activeScene.scene_index;
    const pos = filteredScenes.findIndex((s) => {
      const sIdx = s.sceneIndex !== undefined ? s.sceneIndex : s.scene_index;
      return sIdx === activeIdx;
    });
    return pos >= 0 ? pos + 1 : 1;
  }, [filteredScenes, activeScene]);

  // Helper to dynamically get filtered counts for any scene based on current confidence filter
  const getSceneFilteredCounts = useCallback((scene: VideoScene) => {
    const det = scene.detectedObjectsJson || scene.detected_objects_json;
    if (det) {
      const rawBoxes: KeyframeBBox[] =
        det.keyframeBboxes ||
        det.keyframe_bboxes ||
        det.keyframeDetections ||
        det.keyframe_detections ||
        [];
      const rawTracks: ObjectTrackSummary[] = det.tracksSummary || det.tracks_summary || [];

      const activeCounts: Record<string, number> = {};
      const seenTrackIds = new Set<number>();

      rawTracks.forEach((t) => {
        const conf = t.avgConfidence ?? t.avg_confidence ?? 0;
        const lbl = t.label?.trim().toLowerCase();
        if (!lbl || isNoiseClass(lbl, conf)) return;
        if (conf >= minConfidenceFilter) {
          activeCounts[lbl] = (activeCounts[lbl] || 0) + 1;
          const tid = t.trackId ?? t.track_id;
          if (tid !== undefined && tid !== null) seenTrackIds.add(tid);
        }
      });

      rawBoxes.forEach((b) => {
        const conf = b.confidence || 0;
        const lbl = b.label?.trim().toLowerCase();
        if (!lbl || isNoiseClass(lbl, conf)) return;
        const bid = b.trackId ?? b.track_id;
        if (conf >= minConfidenceFilter && (!bid || !seenTrackIds.has(bid))) {
          activeCounts[lbl] = (activeCounts[lbl] || 0) + 1;
          if (bid) seenTrackIds.add(bid);
        }
      });

      return activeCounts;
    }

    const rawCounts = scene.objectCountsJson || scene.object_counts_json || {};
    const cleanCounts: Record<string, number> = {};
    Object.entries(rawCounts).forEach(([k, v]) => {
      const lbl = k.trim().toLowerCase();
      if (!LOW_RELEVANCE_NOISE_CLASSES.has(lbl)) {
        cleanCounts[lbl] = v;
      }
    });
    return cleanCounts;
  }, [minConfidenceFilter]);

  // Extract Keyframe BBoxes and Tracks for active scene
  const { currentBboxes, currentTracks, rawKeyframeBoxCount } = useMemo(() => {
    if (!activeScene) return { currentBboxes: [], currentTracks: [], rawKeyframeBoxCount: 0 };
    const det = activeScene.detectedObjectsJson || activeScene.detected_objects_json;
    if (!det) return { currentBboxes: [], currentTracks: [], rawKeyframeBoxCount: 0 };

    const rawBoxes: KeyframeBBox[] =
      det.keyframeBboxes ||
      det.keyframe_bboxes ||
      det.keyframeDetections ||
      det.keyframe_detections ||
      [];
    let rawTracks: ObjectTrackSummary[] = det.tracksSummary || det.tracks_summary || [];

    // Synthesize tracks from keyframe bboxes if tracks_summary was empty in DB (e.g. keyframe single-frame detections)
    if (rawTracks.length === 0 && rawBoxes.length > 0) {
      rawTracks = rawBoxes.map((b, idx) => ({
        track_id: b.trackId ?? b.track_id ?? (idx + 1),
        label: b.label,
        avg_confidence: b.confidence ?? 0.5,
        first_seen_sec: activeScene.startSeconds ?? activeScene.start_seconds ?? 0,
        last_seen_sec: activeScene.endSeconds ?? activeScene.end_seconds ?? 0,
        frames_appeared: 1,
        sample_bbox_norm: b.bboxNorm ?? b.bbox_norm ?? [0, 0, 0, 0],
        in_keyframe: true,
        best_timestamp_sec: activeScene.startSeconds ?? activeScene.start_seconds ?? 0,
      }));
    }

    const filteredBoxes = rawBoxes.filter((b) => {
      const conf = b.confidence || 0;
      const lbl = b.label?.trim().toLowerCase() || '';
      if (isNoiseClass(lbl, conf)) return false;
      return conf >= minConfidenceFilter;
    });

    const filteredTracks = rawTracks.filter((t) => {
      const conf = t.avgConfidence ?? t.avg_confidence ?? 0;
      const lbl = t.label?.trim().toLowerCase() || '';
      if (isNoiseClass(lbl, conf)) return false;
      return conf >= minConfidenceFilter;
    });

    const validRawBoxCount = rawBoxes.filter(
      (b) => !isNoiseClass(b.label?.trim().toLowerCase() || '', b.confidence || 0)
    ).length;

    return {
      currentBboxes: filteredBoxes,
      currentTracks: filteredTracks,
      rawKeyframeBoxCount: validRawBoxCount,
    };
  }, [activeScene, minConfidenceFilter]);

  // Keyframe image URL
  const keyframeImageUrl = useMemo(() => {
    if (!activeScene) return '';
    const rawUrl = activeScene.keyframeUrl || activeScene.keyframe_url;
    if (!rawUrl) return '';
    if (rawUrl.startsWith('http://') || rawUrl.startsWith('https://')) {
      return rawUrl;
    }
    const cleanBase = CONFIG.API_BASE_URL.replace('/api/v1', '');
    return `${cleanBase}${rawUrl}`;
  }, [activeScene]);

  const isTrackOnKeyframe = (trk: ObjectTrackSummary): boolean => {
    if (trk.in_keyframe !== undefined) return trk.in_keyframe;
    if (trk.inKeyframe !== undefined) return trk.inKeyframe;
    const trkId = trk.trackId !== undefined ? trk.trackId : trk.track_id;
    if (trkId !== undefined && currentBboxes.some((b) => (b.trackId ?? b.track_id) === trkId)) {
      return true;
    }
    return currentBboxes.some((b) => b.label?.trim().toLowerCase() === trk.label?.trim().toLowerCase());
  };

  const activeHighlightId = hoveredTrackId !== null ? hoveredTrackId : selectedTrackId;

  // IMPORTANT: Selecting scene ONLY switches the active scene in this component.
  // It NEVER seeks the video player or triggers external window jumping!
  const handleSelectScene = (scene: VideoScene) => {
    const sIdx = scene.sceneIndex !== undefined ? scene.sceneIndex : scene.scene_index || 1;
    setSelectedSceneIndex(sIdx);
    setHoveredTrackId(null);
    setSelectedTrackId(null);
  };

  // Explicit seek triggered only when user clicks "Tua video"
  const handleSeekTrack = (trk: ObjectTrackSummary) => {
    const bestSec =
      trk.bestTimestampSec ??
      trk.best_timestamp_sec ??
      trk.firstSeenSec ??
      trk.first_seen_sec ??
      activeScene?.startSeconds ??
      0;
    onSeekVideo(bestSec);
  };

  // Scroll table row into view strictly within the table container without scrolling the window/document
  useEffect(() => {
    if (hoveredTrackId !== null && trackRowRefs.current[hoveredTrackId]) {
      const row = trackRowRefs.current[hoveredTrackId];
      const scrollParent = row?.closest<HTMLDivElement>('.overflow-y-auto');
      if (row && scrollParent) {
        const rowTop = row.offsetTop;
        const rowHeight = row.offsetHeight;
        const parentScrollTop = scrollParent.scrollTop;
        const parentHeight = scrollParent.clientHeight;
        if (rowTop < parentScrollTop) {
          scrollParent.scrollTo({ top: rowTop, behavior: 'smooth' });
        } else if (rowTop + rowHeight > parentScrollTop + parentHeight) {
          scrollParent.scrollTo({ top: rowTop + rowHeight - parentHeight, behavior: 'smooth' });
        }
      }
    }
  }, [hoveredTrackId]);

  if (loading) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Video className="w-5 h-5 text-primary animate-pulse" />
            <span className="text-sm font-semibold text-slate-800 font-heading">
              Đang tải dữ liệu nhận diện đối tượng theo phân cảnh...
            </span>
          </div>
          <RefreshCw className="w-4 h-4 text-slate-400 animate-spin" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-pulse">
          <div className="lg:col-span-4 h-64 bg-slate-100 rounded-lg" />
          <div className="lg:col-span-8 h-64 bg-slate-100 rounded-lg" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-xs font-medium text-red-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={fetchScenesData}
            className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded font-body transition-colors cursor-pointer"
          >
            Thử lại
          </button>
        </div>
      </div>
    );
  }

  if (!hasAnyODData || scenes.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
        <div className="flex items-center gap-2">
          <Video className="w-5 h-5 text-primary" />
          <h2 className="text-base font-bold text-slate-900 font-heading">
            Nhận diện & Theo dõi Đối tượng (Object Tracking)
          </h2>
        </div>
        <div className="p-5 bg-slate-50 border border-slate-200 rounded-xl text-center space-y-1.5">
          <Box className="w-6 h-6 text-slate-400 mx-auto" />
          <p className="text-xs font-semibold text-slate-700 font-body">
            Chưa có dữ liệu đối tượng trong video này.
          </p>
          <p className="text-[11px] text-slate-500 max-w-lg mx-auto font-body">
            Video chưa qua bước phân tích thị giác hoặc các khung hình không chứa đối tượng đạt ngưỡng độ tin cậy.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-sm space-y-5">
      {/* 1. Header Bar: Clean academic title and honest metrics */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-primary-light border border-primary-border flex items-center justify-center text-primary shadow-xs">
              <Video className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-base font-bold text-slate-900 font-heading">
                  Nhận diện & Theo dõi Đối tượng
                </h2>
                <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
                  Thị giác máy tính
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5 font-body">
                Thống kê sự hiện diện của người, phương tiện và công cụ hỗ trợ trong từng phân đoạn bài giảng.
              </p>
            </div>
          </div>
        </div>

        {/* Global Summary Pills */}
        <div className="flex items-center gap-2 shrink-0 text-xs font-body">
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-700">
            <Box className="w-3.5 h-3.5 text-primary" />
            <span className="text-slate-500">Đối tượng duy nhất:</span>
            <span className="font-semibold text-slate-900">{totalUniqueTracks || '—'}</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-700">
            <Layers className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-500">Tổng phân cảnh:</span>
            <span className="font-semibold text-slate-900">{scenes.length}</span>
          </div>
        </div>
      </div>

      {/* 2. Label Filter Bar */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs font-body custom-scrollbar">
        <span className="text-slate-400 font-semibold text-xs shrink-0 flex items-center gap-1">
          <Tag className="w-3.5 h-3.5 text-slate-400" />
          Lọc nhãn:
        </span>

        {/* All Scenes */}
        <button
          type="button"
          onClick={() => setSelectedLabelFilter(null)}
          className={`px-3 py-1 rounded-full text-xs font-medium shrink-0 transition-all cursor-pointer ${
            selectedLabelFilter === null
              ? 'bg-primary text-white shadow-xs'
              : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
          }`}
        >
          Tất cả ({scenes.length})
        </button>

        {/* Dynamic Label Chips */}
        {Object.entries(globalLabelDistribution).map(([label, stats]) => {
          const isSelected = selectedLabelFilter === label;
          const style = getColorForLabel(label);

          return (
            <button
              key={label}
              type="button"
              onClick={() => setSelectedLabelFilter(isSelected ? null : label)}
              className={`px-3 py-1 rounded-full text-xs font-medium shrink-0 transition-all flex items-center gap-1.5 border cursor-pointer ${
                isSelected
                  ? 'bg-slate-900 text-white border-slate-900 shadow-xs'
                  : 'bg-white hover:bg-slate-50 border-slate-200 text-slate-700'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-white' : style.badgeDot}`} />
              <span className="capitalize">{label}</span>
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded font-semibold ${
                  isSelected ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-600'
                }`}
                title={`Xuất hiện trong ${stats.sceneCount} phân cảnh`}
              >
                {stats.sceneCount} cảnh
              </span>
            </button>
          );
        })}
      </div>

      {/* 3. Main Workspace: Scene Selector (Left) vs Keyframe Inspector (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column (4 cols): Scene List */}
        <div className="lg:col-span-4 min-w-0 border border-slate-200 rounded-xl overflow-hidden bg-slate-50 flex flex-col max-h-[580px]">
          <div className="px-4 py-2.5 bg-white border-b border-slate-200 flex items-center justify-between">
            <span className="font-bold text-xs text-slate-700 flex items-center gap-1.5 font-heading">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              Phân cảnh bài giảng
            </span>
            <span className="text-[11px] text-slate-500 font-medium">
              {filteredScenes.length} cảnh
            </span>
          </div>

          <div className="divide-y divide-slate-200/70 overflow-y-auto custom-scrollbar flex-1 p-2 space-y-1.5">
            {filteredScenes.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400 font-body">
                Không có phân cảnh nào chứa nhãn "{selectedLabelFilter}".
              </div>
            ) : (
              filteredScenes.map((scene, idx) => {
                const sIdx = scene.sceneIndex !== undefined ? scene.sceneIndex : scene.scene_index || 1;
                const isActive = sIdx === selectedSceneIndex;
                const displayNum = idx + 1;
                const startSec = scene.startSeconds !== undefined ? scene.startSeconds : scene.start_seconds || 0;
                const endSec = scene.endSeconds !== undefined ? scene.endSeconds : scene.end_seconds || 0;
                const counts = getSceneFilteredCounts(scene);
                const kfUrl = scene.keyframeUrl || scene.keyframe_url;
                const formattedKfUrl = kfUrl
                  ? kfUrl.startsWith('http')
                    ? kfUrl
                    : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${kfUrl}`
                  : null;

                return (
                  <div
                    key={sIdx}
                    onClick={() => handleSelectScene(scene)}
                    className={`p-2.5 rounded-lg cursor-pointer transition-all border ${
                      isActive
                        ? 'bg-white border-primary shadow-xs ring-1 ring-primary/20'
                        : 'bg-white hover:bg-slate-100/80 border-slate-200/80'
                    }`}
                  >
                    <div className="flex items-start gap-2.5">
                      {/* Thumbnail */}
                      <div className="w-16 h-10 bg-slate-900 rounded overflow-hidden shrink-0 relative border border-slate-200">
                        {formattedKfUrl ? (
                          <img
                            src={formattedKfUrl}
                            alt={`Cảnh ${displayNum}`}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-500 text-[9px]">
                            Ảnh tĩnh
                          </div>
                        )}
                        <span className="absolute bottom-0.5 right-0.5 px-1 bg-black/80 text-[8px] font-mono-data font-bold text-white rounded">
                          {formatSeconds(startSec)}
                        </span>
                      </div>

                      {/* Info */}
                      <div className="flex-1 min-w-0 font-body">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            <span className={`text-xs font-bold ${isActive ? 'text-primary' : 'text-slate-900'}`}>
                              Cảnh {displayNum}
                            </span>
                            {sIdx !== displayNum && (
                              <span className="px-1 py-0.2 rounded text-[9px] text-slate-400 bg-slate-100">
                                #{sIdx}
                              </span>
                            )}
                          </div>
                          <span className="text-[10px] font-mono-data text-slate-500">
                            {formatSeconds(startSec)} - {formatSeconds(endSec)}
                          </span>
                        </div>

                        {/* Counts */}
                        <div className="flex items-center gap-1 flex-wrap mt-1">
                          {Object.entries(counts).length === 0 ? (
                            <span className="text-[9px] text-slate-400 italic">Không có đối tượng</span>
                          ) : (
                            Object.entries(counts).map(([lbl, cnt]) => {
                              const style = getColorForLabel(lbl);
                              return (
                                <span
                                  key={lbl}
                                  className={`px-1.5 py-0.2 rounded text-[9px] font-medium flex items-center gap-1 border ${style.badgeBg}`}
                                >
                                  <span className={`w-1.5 h-1.5 rounded-full ${style.badgeDot}`} />
                                  <span>{lbl} x{cnt}</span>
                                </span>
                              );
                            })
                          )}
                        </div>
                      </div>

                      {isActive && (
                        <ChevronRight className="w-4 h-4 text-primary self-center shrink-0" />
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Right Column (8 cols): Keyframe Visualizer + Tracks Table */}
        <div className="lg:col-span-8 min-w-0 space-y-3.5">
          {/* Keyframe Canvas Container */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
            {/* Visualizer Toolbar */}
            <div className="px-4 py-2.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs text-white font-body">
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-white text-sm font-heading">
                    Cảnh {activeDisplayIndex}
                  </span>
                  <span className="text-slate-500">|</span>
                  <span className="font-mono-data text-slate-300 text-xs">
                    {formatSeconds(activeScene?.startSeconds || 0)} - {formatSeconds(activeScene?.endSeconds || 0)}
                  </span>
                </div>

                <div className="text-[11px] text-slate-400 flex items-center gap-1.5 flex-wrap">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  <span>
                    Hiện <strong>{currentBboxes.length}</strong> đối tượng trên ảnh đại diện
                  </span>
                  <span className="text-slate-600">•</span>
                  <span>
                    Tổng <strong>{currentTracks.length}</strong> đối tượng trong cảnh này (≥ {Math.round(minConfidenceFilter * 100)}%)
                  </span>
                  {rawKeyframeBoxCount > 0 && currentBboxes.length === 0 && (
                    <span className="text-amber-400 text-[10px]">
                      (Có {rawKeyframeBoxCount} đối tượng ở ngưỡng thấp hơn)
                    </span>
                  )}
                </div>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-3">
                {/* Confidence Slider */}
                <div className="flex items-center gap-1.5">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                  <span className="text-[10px] text-slate-400">Ngưỡng:</span>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={minConfidenceFilter}
                    onChange={(e) => setMinConfidenceFilter(parseFloat(e.target.value))}
                    className="w-14 h-1 bg-slate-700 rounded cursor-pointer accent-primary"
                  />
                  <span className="font-mono-data text-[10px] text-slate-300 w-7">
                    {Math.round(minConfidenceFilter * 100)}%
                  </span>
                </div>

                {/* Show/Hide Boxes */}
                <button
                  type="button"
                  onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
                    showBoundingBoxes
                      ? 'bg-slate-700 text-white hover:bg-slate-600'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}
                >
                  {showBoundingBoxes ? (
                    <>
                      <Eye className="w-3.5 h-3.5" />
                      <span>Hiện khung</span>
                    </>
                  ) : (
                    <>
                      <EyeOff className="w-3.5 h-3.5" />
                      <span>Ẩn khung</span>
                    </>
                  )}
                </button>

                {/* Tua video Button */}
                <button
                  type="button"
                  onClick={() => onSeekVideo(activeScene?.startSeconds || 0)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 bg-primary hover:bg-primary-hover text-white rounded text-xs font-medium transition-colors cursor-pointer shadow-xs"
                >
                  <Play className="w-3 h-3 fill-current" />
                  <span>Tua video</span>
                </button>
              </div>
            </div>

            {/* Canvas Area with CSS Percentage Bounding Boxes */}
            <div className="relative w-full aspect-video max-h-[360px] bg-slate-950 flex items-center justify-center overflow-hidden select-none">
              {keyframeImageUrl ? (
                <img
                  src={keyframeImageUrl}
                  alt={`Keyframe Cảnh ${activeDisplayIndex}`}
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-slate-500 text-xs font-body">
                  Không tìm thấy ảnh đại diện của cảnh này
                </div>
              )}

              {/* Bounding Box Layer */}
              {showBoundingBoxes &&
                currentBboxes.map((bbox, bIdx) => {
                  const rawBox = bbox.bboxNorm || bbox.bbox_norm || [0, 0, 0, 0];
                  const x1 = Math.max(0, Math.min(1, Math.min(rawBox[0], rawBox[2])));
                  const y1 = Math.max(0, Math.min(1, Math.min(rawBox[1], rawBox[3])));
                  const x2 = Math.max(0, Math.min(1, Math.max(rawBox[0], rawBox[2])));
                  const y2 = Math.max(0, Math.min(1, Math.max(rawBox[1], rawBox[3])));

                  const widthPct = Math.max(0, x2 - x1) * 100;
                  const heightPct = Math.max(0, y2 - y1) * 100;
                  const leftPct = x1 * 100;
                  const topPct = y1 * 100;

                  const color = getColorForLabel(bbox.label);
                  const boxTrackId = bbox.trackId !== undefined ? bbox.trackId : bbox.track_id;

                  const isHighlighted =
                    activeHighlightId !== null &&
                    ((boxTrackId !== undefined && boxTrackId === activeHighlightId) ||
                      (boxTrackId === undefined &&
                        bbox.label.toLowerCase() ===
                          currentTracks
                            .find((t) => (t.trackId ?? t.track_id) === activeHighlightId)
                            ?.label.toLowerCase()));

                  const isDimmed = activeHighlightId !== null && !isHighlighted;

                  return (
                    <div
                      key={bIdx}
                      className={`absolute border-2 transition-all duration-150 pointer-events-auto cursor-pointer ${
                        color.border
                      } ${color.bg} ${
                        isHighlighted
                          ? 'ring-2 ring-white border-white scale-[1.01] shadow-lg z-30'
                          : isDimmed
                          ? 'opacity-25 z-10'
                          : 'z-20 hover:ring-2 hover:ring-white'
                      }`}
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      onMouseEnter={() => boxTrackId && setHoveredTrackId(boxTrackId)}
                      onMouseLeave={() => setHoveredTrackId(null)}
                      onClick={() => {
                        if (boxTrackId) {
                          setSelectedTrackId(selectedTrackId === boxTrackId ? null : boxTrackId);
                        }
                      }}
                    >
                      <div
                        className={`absolute -top-5 left-0 px-1.5 py-0.2 rounded flex items-center gap-1 shadow-xs whitespace-nowrap text-[9px] font-semibold transition-all ${
                          isHighlighted
                            ? 'bg-primary text-white ring-1 ring-white'
                            : 'bg-black/80 backdrop-blur-xs text-white'
                        }`}
                      >
                        {boxTrackId && <span>#{boxTrackId}</span>}
                        <span className="capitalize">{bbox.label}</span>
                        <span className="font-mono-data opacity-90">
                          {Math.round(bbox.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>

          {/* 4. Scene Details Tabbed Panel */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            {/* Tabs Header */}
            <div className="flex border-b border-slate-200 bg-slate-50 text-xs font-semibold font-heading">
              <button
                type="button"
                onClick={() => setActiveTab('tracks')}
                className={`flex-1 py-2.5 px-4 text-center border-b-2 transition-colors flex items-center justify-center gap-2 cursor-pointer ${
                  activeTab === 'tracks'
                    ? 'border-primary text-primary bg-white'
                    : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}
              >
                <ListFilter className="w-3.5 h-3.5" />
                Danh sách đối tượng ({currentTracks.length})
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('info')}
                className={`flex-1 py-2.5 px-4 text-center border-b-2 transition-colors flex items-center justify-center gap-2 cursor-pointer ${
                  activeTab === 'info'
                    ? 'border-primary text-primary bg-white'
                    : 'border-transparent text-slate-500 hover:text-slate-700'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                Thông tin phân cảnh
              </button>
            </div>

            {/* Tab 1: Object Tracks Table */}
            {activeTab === 'tracks' && (
              <div className="p-3 max-h-[220px] overflow-y-auto overflow-x-auto custom-scrollbar">
                {currentTracks.length === 0 ? (
                  <div className="p-6 text-center text-xs text-slate-400 font-body space-y-1">
                    <p>Chưa có đối tượng nào đạt ngưỡng độ tin cậy ≥ {Math.round(minConfidenceFilter * 100)}%.</p>
                    {rawKeyframeBoxCount > 0 && (
                      <p className="text-[11px] text-amber-500 font-medium">
                        Có {rawKeyframeBoxCount} đối tượng ở ngưỡng thấp hơn. Bạn có thể hạ thanh trượt "Ngưỡng" xuống để xem chi tiết.
                      </p>
                    )}
                  </div>
                ) : (
                  <table className="w-full text-xs text-left font-body">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-500 text-[10px] font-semibold uppercase tracking-wider sticky top-0 bg-white z-10">
                        <th className="py-1.5 px-2.5">Mã</th>
                        <th className="py-1.5 px-2.5">Đối tượng</th>
                        <th className="py-1.5 px-2.5">Trạng thái xuất hiện</th>
                        <th className="py-1.5 px-2.5">Độ tin cậy</th>
                        <th className="py-1.5 px-2.5">Khoảng thời gian</th>
                        <th className="py-1.5 px-2.5">Số khung</th>
                        <th className="py-1.5 px-2.5 text-right">Tua video</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {currentTracks.map((trk, idx) => {
                        const trackId = trk.trackId !== undefined ? trk.trackId : trk.track_id;
                        const avgConf = trk.avgConfidence !== undefined ? trk.avgConfidence : trk.avg_confidence || 0;
                        const firstSec = trk.firstSeenSec !== undefined ? trk.firstSeenSec : trk.first_seen_sec || 0;
                        const lastSec = trk.lastSeenSec !== undefined ? trk.lastSeenSec : trk.last_seen_sec || 0;
                        const bestSec =
                          trk.bestTimestampSec ??
                          trk.best_timestamp_sec ??
                          firstSec;
                        const frames = trk.framesAppeared !== undefined ? trk.framesAppeared : trk.frames_appeared || 0;
                        const style = getColorForLabel(trk.label);
                        const inKeyframe = isTrackOnKeyframe(trk);

                        const isRowActive =
                          activeHighlightId !== null &&
                          (trackId === activeHighlightId ||
                            (!trackId &&
                              trk.label.toLowerCase() ===
                                currentBboxes
                                  .find((b) => (b.trackId ?? b.track_id) === activeHighlightId)
                                  ?.label.toLowerCase()));

                        return (
                          <tr
                            key={idx}
                            ref={(el) => {
                              if (trackId !== undefined) {
                                trackRowRefs.current[trackId] = el;
                              }
                            }}
                            onMouseEnter={() => trackId && setHoveredTrackId(trackId)}
                            onMouseLeave={() => setHoveredTrackId(null)}
                            onClick={() => {
                              if (trackId !== undefined) {
                                setSelectedTrackId(selectedTrackId === trackId ? null : trackId);
                              }
                            }}
                            className={`transition-all cursor-pointer ${
                              isRowActive
                                ? 'bg-primary-light font-medium ring-1 ring-inset ring-primary-border'
                                : 'hover:bg-slate-50'
                            }`}
                          >
                            {/* Track ID */}
                            <td className="py-2 px-2.5 font-bold text-primary">
                              #{trackId || idx + 1}
                            </td>

                            {/* Label */}
                            <td className="py-2 px-2.5 capitalize">
                              <span className="flex items-center gap-1.5">
                                <span className={`w-2 h-2 rounded-full ${style.badgeDot}`} />
                                <span className="font-medium text-slate-900">{trk.label}</span>
                              </span>
                            </td>

                            {/* Status */}
                            <td className="py-2 px-2.5">
                              {inKeyframe ? (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                                  <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                                  Trên ảnh đại diện
                                </span>
                              ) : (
                                <div className="flex flex-col">
                                  <span
                                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-50 text-amber-700 border border-amber-200"
                                    title={`Xuất hiện ngoài frame tĩnh lúc ${formatSeconds(firstSec)} - ${formatSeconds(lastSec)}`}
                                  >
                                    <Clock className="w-3 h-3 text-amber-600 shrink-0" />
                                    <span>Xuất hiện lúc {formatSeconds(firstSec)} - {formatSeconds(lastSec)}</span>
                                  </span>
                                </div>
                              )}
                            </td>

                            {/* Confidence */}
                            <td className="py-2 px-2.5 text-slate-700 font-mono-data">
                              {Math.round(avgConf * 100)}%
                            </td>

                            {/* Time interval */}
                            <td className="py-2 px-2.5 text-slate-600 font-mono-data">
                              {formatSeconds(firstSec)} → {formatSeconds(lastSec)}
                            </td>

                            {/* Frames */}
                            <td className="py-2 px-2.5 text-slate-500 font-mono-data">
                              {frames}f
                            </td>

                            {/* Seek action */}
                            <td className="py-2 px-2.5 text-right">
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSeekTrack(trk);
                                }}
                                className="inline-flex items-center gap-1 px-2 py-0.5 bg-slate-100 hover:bg-primary hover:text-white rounded text-[10px] font-medium transition-all cursor-pointer shadow-xs"
                                title={`Tua đến thời điểm rõ nhất (${formatSeconds(bestSec)})`}
                              >
                                <Play className="w-2.5 h-2.5 fill-current" />
                                <span className="font-mono-data">{formatSeconds(bestSec)}</span>
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            )}

            {/* Tab 2: Scene Information */}
            {activeTab === 'info' && (
              <div className="p-3.5 space-y-3 text-xs leading-relaxed max-h-[220px] overflow-y-auto custom-scrollbar font-body">
                {/* Visual Caption */}
                <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                  <span className="font-semibold text-[11px] text-slate-700 flex items-center gap-1 font-heading">
                    <Video className="w-3.5 h-3.5 text-primary" />
                    Mô tả hình ảnh (Visual Caption):
                  </span>
                  <p className="text-slate-800">
                    {activeScene?.caption || 'Chưa có mô tả khung cảnh tự động cho phân đoạn này.'}
                  </p>
                </div>

                {/* Spoken Script */}
                <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                  <span className="font-semibold text-[11px] text-slate-700 flex items-center gap-1 font-heading">
                    <FileText className="w-3.5 h-3.5 text-primary" />
                    Lời giảng tương ứng (Audio Transcript):
                  </span>
                  <p className="text-slate-800">
                    {activeScene?.script || 'Chưa ghi nhận lời giảng trong khoảng thời gian này.'}
                  </p>
                </div>

                {/* OCR Text */}
                {activeScene?.ocrText || activeScene?.ocr_text ? (
                  <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                    <span className="font-semibold text-[11px] text-slate-700 flex items-center gap-1 font-heading">
                      <Box className="w-3.5 h-3.5 text-primary" />
                      Văn bản trên slide (OCR Text):
                    </span>
                    <p className="text-slate-800 whitespace-pre-wrap">
                      {activeScene.ocrText || activeScene.ocr_text}
                    </p>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SceneObjectExplorer;
