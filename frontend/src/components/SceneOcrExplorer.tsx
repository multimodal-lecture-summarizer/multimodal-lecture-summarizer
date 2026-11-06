/* SceneOcrExplorer: shows stored PaddleOCR output per scene (text, confidence, boxes).
 * Mirrors SceneObjectExplorer: scene list (left), keyframe with box overlay + tabbed details (right),
 * confidence threshold, 2-way hover/select sync between boxes and table rows.
 */

import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  ScanText,
  Layers,
  Search,
  Clock,
  ChevronRight,
  SlidersHorizontal,
  Eye,
  EyeOff,
  Play,
  Copy,
  Check,
  Download,
  AlertCircle,
  ListFilter,
  FileText,
  BarChart3,
} from 'lucide-react';
import { api } from '../services/api';
import { formatSeconds, resolveKeyframeUrl } from '../utils/scene-format';
import {
  normalizeOcrScene,
  confidenceBand,
  highlightParts,
  computeOcrStats,
  buildSceneText,
  countWords,
  type OcrScene,
  type NormBlock,
  type ConfidenceBand,
} from '../utils/ocr-blocks';

interface SceneOcrExplorerProps {
  videoId: string;
  onSeekVideo: (seconds: number) => void;
}

const BAND_STYLE: Record<ConfidenceBand, { color: string; badge: string; bar: string; label: string }> = {
  high: { color: '#10b981', badge: 'bg-emerald-50 text-emerald-700 border-emerald-200', bar: 'bg-emerald-500', label: '≥ 90%' },
  mid: { color: '#f59e0b', badge: 'bg-amber-50 text-amber-700 border-amber-200', bar: 'bg-amber-500', label: '75–90%' },
  low: { color: '#f43f5e', badge: 'bg-rose-50 text-rose-700 border-rose-200', bar: 'bg-rose-500', label: '< 75%' },
};

const HIST_LABELS = ['50–60', '60–70', '70–80', '80–90', '90–100'];

type FilterMode = 'all' | 'with' | 'without';
type TabId = 'blocks' | 'text' | 'stats';

const Highlighted: React.FC<{ text: string; query: string }> = ({ text, query }) => (
  <>
    {highlightParts(text, query).map((p, i) =>
      p.match ? (
        <mark key={i} className="bg-yellow-200 text-slate-900 rounded-sm px-0.5">{p.text}</mark>
      ) : (
        <React.Fragment key={i}>{p.text}</React.Fragment>
      )
    )}
  </>
);

export const SceneOcrExplorer: React.FC<SceneOcrExplorerProps> = ({ videoId, onSeekVideo }) => {
  const [scenes, setScenes] = useState<OcrScene[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [minConf, setMinConf] = useState(0.5);
  const [search, setSearch] = useState('');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [activeTab, setActiveTab] = useState<TabId>('blocks');
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const rowRefs = useRef<Record<number, HTMLTableRowElement | null>>({});

  const fetchScenes = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getVideoScenes(videoId);
      if (res.success && Array.isArray(res.data)) {
        const list = (res.data as any[]).map(normalizeOcrScene);
        setScenes(list);
        if (list.length > 0) {
          setSelectedIdx((list.find((s) => s.blocks.length > 0) ?? list[0]).index);
        }
      } else {
        throw new Error(res.message || 'Không thể tải danh sách phân cảnh bài giảng.');
      }
    } catch (err: any) {
      setError(err.message || 'Lỗi kết nối khi tải dữ liệu OCR.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (videoId) fetchScenes();
  }, [videoId]);

  const query = search.trim().toLowerCase();
  const visibleOf = useCallback(
    (s: OcrScene) => s.blocks.filter((b) => b.confidence >= minConf),
    [minConf]
  );
  const isMatch = (b: NormBlock) => !!query && b.text.toLowerCase().includes(query);

  const hasAnyData = useMemo(() => scenes.some((s) => s.blocks.length > 0 || s.ocrText), [scenes]);
  const hasAnyBlocks = useMemo(() => scenes.some((s) => s.blocks.length > 0), [scenes]);

  // Header totals across all scenes at the current threshold
  const totals = useMemo(() => {
    let withText = 0;
    let blocks = 0;
    let words = 0;
    let confSum = 0;
    scenes.forEach((s) => {
      const vis = s.blocks.filter((b) => b.confidence >= minConf);
      if (vis.length) withText += 1;
      vis.forEach((b) => {
        blocks += 1;
        words += countWords(b.text);
        confSum += b.confidence;
      });
    });
    return { withText, blocks, words, avg: blocks ? confSum / blocks : 0 };
  }, [scenes, minConf]);

  const filteredScenes = useMemo(
    () =>
      scenes.filter((s) => {
        const vis = visibleOf(s);
        if (filterMode === 'with' && vis.length === 0) return false;
        if (filterMode === 'without' && vis.length > 0) return false;
        if (query && !vis.some((b) => b.text.toLowerCase().includes(query))) return false;
        return true;
      }),
    [scenes, visibleOf, filterMode, query]
  );

  const activeScene = useMemo(() => {
    const found = filteredScenes.find((s) => s.index === selectedIdx);
    return found ?? filteredScenes[0] ?? null;
  }, [filteredScenes, selectedIdx]);

  const activeBlocks = useMemo(() => (activeScene ? visibleOf(activeScene) : []), [activeScene, visibleOf]);
  const stats = useMemo(
    () => computeOcrStats(activeScene?.blocks ?? [], minConf),
    [activeScene, minConf]
  );
  const imageUrl = resolveKeyframeUrl(activeScene?.keyframeUrl);
  const activeLine = hoveredLine ?? selectedLine;

  useEffect(() => {
    if (hoveredLine !== null && activeTab === 'blocks') {
      rowRefs.current[hoveredLine]?.scrollIntoView({ block: 'nearest' });
    }
  }, [hoveredLine, activeTab]);

  const selectScene = (s: OcrScene) => {
    setSelectedIdx(s.index);
    setHoveredLine(null);
    setSelectedLine(null);
  };

  const copy = async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey((k) => (k === key ? null : k)), 1500);
    } catch {
      /* clipboard unavailable (insecure context): ignore */
    }
  };

  const exportJson = () => {
    const payload = scenes.map((s) => ({
      sceneIndex: s.index,
      startSeconds: s.startSec,
      endSeconds: s.endSec,
      ocrText: s.ocrText,
      blocks: s.blocks,
    }));
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr-${videoId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleLine = (line: number) => setSelectedLine((cur) => (cur === line ? null : line));

  // ---------- states ----------
  if (loading) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm animate-pulse space-y-3">
        <div className="h-5 w-64 bg-slate-100 rounded" />
        <div className="h-40 bg-slate-50 rounded" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-xs font-medium text-red-700 flex items-center justify-between">
          <span className="flex items-center gap-2"><AlertCircle className="w-4 h-4 shrink-0" />{error}</span>
          <button type="button" onClick={fetchScenes} className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded cursor-pointer">
            Thử lại
          </button>
        </div>
      </div>
    );
  }

  const header = (
    <div className="flex items-center gap-2.5">
      <div className="w-8 h-8 rounded-lg bg-primary-light border border-primary-border flex items-center justify-center text-primary shadow-xs">
        <ScanText className="w-4 h-4" />
      </div>
      <div>
        <div className="flex items-center gap-2 flex-wrap">
          <h2 className="text-base font-bold text-slate-900 font-heading">Nhận dạng chữ trên slide (OCR)</h2>
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700 border border-slate-200">PaddleOCR</span>
        </div>
        <p className="text-xs text-slate-500 mt-0.5 font-body">
          Văn bản, độ tin cậy và vị trí từng dòng chữ được nhận dạng trên ảnh đại diện của mỗi phân cảnh.
        </p>
      </div>
    </div>
  );

  if (!hasAnyData || scenes.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
        {header}
        <div className="p-5 bg-slate-50 border border-slate-200 rounded-xl text-center space-y-1.5">
          <ScanText className="w-6 h-6 text-slate-400 mx-auto" />
          <p className="text-xs font-semibold text-slate-700 font-body">Chưa có dữ liệu OCR chi tiết cho video này.</p>
          <p className="text-[11px] text-slate-500 max-w-lg mx-auto font-body">
            Video được xử lý trước khi hệ thống lưu kết quả OCR, hoặc các khung hình không chứa chữ. Hãy xử lý lại video để xem.
          </p>
        </div>
      </div>
    );
  }

  const pill = (icon: React.ReactNode, label: string, value: string) => (
    <div className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-700">
      {icon}
      <span className="text-slate-500">{label}</span>
      <span className="font-semibold text-slate-900">{value}</span>
    </div>
  );

  const chip = (mode: FilterMode, label: string) => (
    <button
      key={mode}
      type="button"
      onClick={() => setFilterMode(mode)}
      className={`px-3 py-1 rounded-full text-xs font-medium shrink-0 transition-all cursor-pointer ${
        filterMode === mode ? 'bg-primary text-white shadow-xs' : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
      }`}
    >
      {label}
    </button>
  );

  const tabBtn = (id: TabId, icon: React.ReactNode, label: string) => (
    <button
      type="button"
      onClick={() => setActiveTab(id)}
      className={`flex-1 py-2.5 px-3 text-center border-b-2 transition-colors flex items-center justify-center gap-2 cursor-pointer ${
        activeTab === id ? 'border-primary text-primary bg-white' : 'border-transparent text-slate-500 hover:text-slate-700'
      }`}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 shadow-sm space-y-5">
      {/* Header + totals */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
        {header}
        <div className="flex items-center gap-2 flex-wrap text-xs font-body">
          {pill(<Layers className="w-3.5 h-3.5 text-primary" />, 'Cảnh có chữ:', `${totals.withText}/${scenes.length}`)}
          {pill(<ScanText className="w-3.5 h-3.5 text-slate-500" />, 'Dòng:', String(totals.blocks))}
          {pill(<FileText className="w-3.5 h-3.5 text-slate-500" />, 'Từ:', String(totals.words))}
          {pill(<BarChart3 className="w-3.5 h-3.5 text-slate-500" />, 'Tin cậy TB:', totals.blocks ? `${Math.round(totals.avg * 100)}%` : '—')}
          {hasAnyBlocks && (
            <button
              type="button"
              onClick={exportJson}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 font-medium cursor-pointer"
              title="Tải toàn bộ kết quả OCR (JSON)"
            >
              <Download className="w-3.5 h-3.5" /> JSON
            </button>
          )}
        </div>
      </div>

      {/* Search + filter chips */}
      <div className="flex items-center gap-2 flex-wrap text-xs font-body">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Tìm chữ trong tất cả slide..."
            aria-label="Tìm chữ trong kết quả OCR"
            className="pl-8 pr-3 py-1.5 w-64 border border-slate-200 rounded-full text-xs focus:outline-none focus:ring-2 focus:ring-primary/30"
          />
        </div>
        {query && (
          <span className="text-slate-500">{filteredScenes.length} cảnh khớp</span>
        )}
        <span className="mx-1 text-slate-300">|</span>
        {chip('all', `Tất cả (${scenes.length})`)}
        {chip('with', 'Có chữ')}
        {chip('without', 'Không có chữ')}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Scene list */}
        <div className="lg:col-span-4 min-w-0 border border-slate-200 rounded-xl overflow-hidden bg-slate-50 flex flex-col max-h-[580px]">
          <div className="px-4 py-2.5 bg-white border-b border-slate-200 flex items-center justify-between">
            <span className="font-bold text-xs text-slate-700 flex items-center gap-1.5 font-heading">
              <Clock className="w-3.5 h-3.5 text-slate-400" /> Phân cảnh bài giảng
            </span>
            <span className="text-[11px] text-slate-500 font-medium">{filteredScenes.length} cảnh</span>
          </div>
          <div className="overflow-y-auto custom-scrollbar flex-1 p-2 space-y-1.5">
            {filteredScenes.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400 font-body">Không có phân cảnh nào phù hợp bộ lọc.</div>
            ) : (
              filteredScenes.map((s, i) => {
                const vis = visibleOf(s);
                const isActive = s.index === activeScene?.index;
                const avg = vis.length ? vis.reduce((a, b) => a + b.confidence, 0) / vis.length : 0;
                const snippet = vis.map((b) => b.text).join(' · ');
                const thumb = resolveKeyframeUrl(s.keyframeUrl);
                return (
                  <div
                    key={s.index}
                    onClick={() => selectScene(s)}
                    className={`p-2.5 rounded-lg cursor-pointer transition-all border ${
                      isActive ? 'bg-white border-primary shadow-xs ring-1 ring-primary/20' : 'bg-white hover:bg-slate-100/80 border-slate-200/80'
                    }`}
                  >
                    <div className="flex items-start gap-2.5">
                      <div className="w-16 h-10 bg-slate-900 rounded overflow-hidden shrink-0 relative border border-slate-200">
                        {thumb ? (
                          <img src={thumb} alt={`Cảnh ${i + 1}`} className="w-full h-full object-cover" loading="lazy" />
                        ) : null}
                        <span className="absolute bottom-0.5 right-0.5 px-1 bg-black/80 text-[8px] font-mono-data font-bold text-white rounded">
                          {formatSeconds(s.startSec)}
                        </span>
                      </div>
                      <div className="flex-1 min-w-0 font-body">
                        <div className="flex items-center justify-between">
                          <span className={`text-xs font-bold ${isActive ? 'text-primary' : 'text-slate-900'}`}>
                            Cảnh {s.index}
                          </span>
                          <span className="text-[10px] font-mono-data text-slate-500">
                            {formatSeconds(s.startSec)} - {formatSeconds(s.endSec)}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 mt-1">
                          {vis.length === 0 ? (
                            <span className="text-[9px] text-slate-400 italic">Không có chữ</span>
                          ) : (
                            <>
                              <span className="px-1.5 rounded text-[9px] font-medium border bg-slate-50 text-slate-700 border-slate-200">
                                {vis.length} dòng
                              </span>
                              <span className={`px-1.5 rounded text-[9px] font-medium border ${BAND_STYLE[confidenceBand(avg)].badge}`}>
                                {Math.round(avg * 100)}%
                              </span>
                            </>
                          )}
                        </div>
                        {snippet && (
                          <p className="text-[10px] text-slate-500 mt-1 line-clamp-2 break-words">
                            <Highlighted text={snippet.slice(0, 120)} query={search} />
                          </p>
                        )}
                      </div>
                      {isActive && <ChevronRight className="w-4 h-4 text-primary self-center shrink-0" />}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Inspector */}
        <div className="lg:col-span-8 min-w-0 space-y-3.5">
          {!activeScene ? (
            <div className="p-8 text-center text-xs text-slate-400 border border-slate-200 rounded-xl">Chọn một phân cảnh để xem chi tiết.</div>
          ) : (
            <>
              <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
                <div className="px-4 py-2.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs text-white font-body">
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm font-heading">Cảnh {activeScene.index}</span>
                      <span className="text-slate-500">|</span>
                      <span className="font-mono-data text-slate-300">
                        {formatSeconds(activeScene.startSec)} - {formatSeconds(activeScene.endSec)}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 flex items-center gap-1.5 flex-wrap">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <span>Hiện <strong>{activeBlocks.length}</strong> dòng chữ (≥ {Math.round(minConf * 100)}%)</span>
                      {stats.hidden > 0 && (
                        <span className="text-amber-400 text-[10px]">· {stats.hidden} dòng bị ẩn bởi ngưỡng</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3 flex-wrap">
                    <label className="flex items-center gap-1.5">
                      <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                      <span className="text-[10px] text-slate-400">Ngưỡng:</span>
                      <input
                        type="range"
                        min={0.5}
                        max={0.95}
                        step={0.05}
                        value={minConf}
                        onChange={(e) => setMinConf(parseFloat(e.target.value))}
                        className="w-14 h-1 bg-slate-700 rounded cursor-pointer accent-primary"
                        aria-label="Ngưỡng độ tin cậy OCR"
                      />
                      <span className="font-mono-data text-[10px] text-slate-300 w-7">{Math.round(minConf * 100)}%</span>
                    </label>
                    <button
                      type="button"
                      onClick={() => setShowBoxes(!showBoxes)}
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium cursor-pointer ${
                        showBoxes ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                      }`}
                    >
                      {showBoxes ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                      {showBoxes ? 'Hiện khung' : 'Ẩn khung'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowLabels(!showLabels)}
                      disabled={!showBoxes}
                      className={`px-2 py-1 rounded text-xs font-medium cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                        showLabels ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                      }`}
                    >
                      Nhãn chữ
                    </button>
                    <button
                      type="button"
                      onClick={() => onSeekVideo(activeScene.startSec)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 bg-primary hover:bg-primary-hover text-white rounded text-xs font-medium cursor-pointer shadow-xs"
                    >
                      <Play className="w-3 h-3 fill-current" /> Tua video
                    </button>
                  </div>
                </div>

                {/* Canvas: wrapper takes the image's own size so normalized boxes line up with pixels */}
                <div className="w-full bg-slate-950 flex items-center justify-center overflow-hidden select-none p-1">
                  {imageUrl ? (
                    <div className="relative inline-block max-w-full">
                      <img src={imageUrl} alt={`Keyframe cảnh ${activeScene.index}`} className="block max-w-full max-h-[360px] w-auto h-auto" />
                      {showBoxes && (
                        <>
                          <svg
                            className="absolute inset-0 w-full h-full"
                            viewBox="0 0 1 1"
                            preserveAspectRatio="none"
                            aria-hidden="true"
                          >
                            {activeBlocks.map((b) => {
                              const band = BAND_STYLE[confidenceBand(b.confidence)];
                              const on = activeLine === b.lineIndex;
                              const dim = (activeLine !== null && !on) || (!!query && !isMatch(b));
                              return (
                                <polygon
                                  key={b.lineIndex}
                                  points={b.polygon.map((p) => `${p[0]},${p[1]}`).join(' ')}
                                  fill={band.color}
                                  fillOpacity={on ? 0.35 : 0.15}
                                  stroke={on ? '#ffffff' : band.color}
                                  strokeWidth={on ? 3 : 2}
                                  vectorEffect="non-scaling-stroke"
                                  opacity={dim ? 0.25 : 1}
                                  className="cursor-pointer transition-opacity"
                                  style={{ pointerEvents: 'all' }}
                                  onMouseEnter={() => setHoveredLine(b.lineIndex)}
                                  onMouseLeave={() => setHoveredLine(null)}
                                  onClick={() => toggleLine(b.lineIndex)}
                                />
                              );
                            })}
                          </svg>
                          {activeBlocks.map((b) => {
                            const on = activeLine === b.lineIndex;
                            if (!on && !(showLabels && (!query || isMatch(b)))) return null;
                            const text = b.text.length > 24 ? `${b.text.slice(0, 24)}…` : b.text;
                            return (
                              <div
                                key={b.lineIndex}
                                className={`absolute pointer-events-none px-1 rounded whitespace-nowrap text-[9px] font-semibold ${
                                  on ? 'bg-primary text-white ring-1 ring-white z-30' : 'bg-black/80 text-white z-20'
                                }`}
                                style={{ left: `${b.bbox[0] * 100}%`, top: `${b.bbox[1] * 100}%`, transform: 'translateY(-100%)' }}
                              >
                                {on ? b.text : text} <span className="font-mono-data opacity-80">{Math.round(b.confidence * 100)}%</span>
                              </div>
                            );
                          })}
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="text-slate-500 text-xs font-body py-16">Không tìm thấy ảnh đại diện của cảnh này</div>
                  )}
                </div>

                {/* Legend */}
                <div className="px-4 py-1.5 bg-slate-900 border-t border-slate-800 flex items-center gap-4 text-[10px] text-slate-400">
                  <span>Độ tin cậy:</span>
                  {(['high', 'mid', 'low'] as ConfidenceBand[]).map((k) => (
                    <span key={k} className="flex items-center gap-1">
                      <span className="w-2.5 h-2.5 rounded-sm" style={{ background: BAND_STYLE[k].color }} />
                      {BAND_STYLE[k].label}
                    </span>
                  ))}
                </div>
              </div>

              {/* Tabs */}
              <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
                <div className="flex border-b border-slate-200 bg-slate-50 text-xs font-semibold font-heading">
                  {tabBtn('blocks', <ListFilter className="w-3.5 h-3.5" />, `Khối văn bản (${activeBlocks.length})`)}
                  {tabBtn('text', <FileText className="w-3.5 h-3.5" />, 'Văn bản đầy đủ')}
                  {tabBtn('stats', <BarChart3 className="w-3.5 h-3.5" />, 'Thống kê')}
                </div>

                {activeTab === 'blocks' && (
                  <div className="p-3 max-h-[260px] overflow-y-auto overflow-x-auto custom-scrollbar">
                    {activeBlocks.length === 0 ? (
                      <div className="p-6 text-center text-xs text-slate-400 font-body space-y-1">
                        <p>Cảnh này không có chữ đạt ngưỡng ≥ {Math.round(minConf * 100)}%.</p>
                        {stats.hidden > 0 && (
                          <p className="text-[11px] text-amber-500 font-medium">Có {stats.hidden} dòng ở ngưỡng thấp hơn; hạ thanh trượt để xem.</p>
                        )}
                      </div>
                    ) : (
                      <table className="w-full text-xs text-left font-body">
                        <thead>
                          <tr className="border-b border-slate-200 text-slate-500 text-[10px] font-semibold uppercase tracking-wider sticky top-0 bg-white z-10">
                            <th className="py-1.5 px-2.5">#</th>
                            <th className="py-1.5 px-2.5">Văn bản</th>
                            <th className="py-1.5 px-2.5">Độ tin cậy</th>
                            <th className="py-1.5 px-2.5">Vị trí (x,y)</th>
                            <th className="py-1.5 px-2.5">Kích thước</th>
                            <th className="py-1.5 px-2.5">Ký tự</th>
                            <th className="py-1.5 px-2.5 text-right">Chép</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-slate-700">
                          {activeBlocks.map((b) => {
                            const band = BAND_STYLE[confidenceBand(b.confidence)];
                            const on = activeLine === b.lineIndex;
                            const key = `row-${b.lineIndex}`;
                            return (
                              <tr
                                key={b.lineIndex}
                                ref={(el) => { rowRefs.current[b.lineIndex] = el; }}
                                onMouseEnter={() => setHoveredLine(b.lineIndex)}
                                onMouseLeave={() => setHoveredLine(null)}
                                onClick={() => toggleLine(b.lineIndex)}
                                className={`transition-all cursor-pointer ${
                                  on ? 'bg-primary-light font-medium ring-1 ring-inset ring-primary-border' : 'hover:bg-slate-50'
                                } ${query && !isMatch(b) ? 'opacity-50' : ''}`}
                              >
                                <td className="py-2 px-2.5 font-bold text-primary">{b.lineIndex + 1}</td>
                                <td className="py-2 px-2.5 text-slate-900 break-words max-w-[260px]">
                                  <Highlighted text={b.text} query={search} />
                                </td>
                                <td className="py-2 px-2.5">
                                  <div className="flex items-center gap-2">
                                    <div className="w-14 h-1.5 bg-slate-100 rounded overflow-hidden">
                                      <div className={`h-full ${band.bar}`} style={{ width: `${Math.round(b.confidence * 100)}%` }} />
                                    </div>
                                    <span className="font-mono-data">{(b.confidence * 100).toFixed(1)}%</span>
                                  </div>
                                </td>
                                <td className="py-2 px-2.5 font-mono-data text-slate-600">
                                  {(b.bbox[0] * 100).toFixed(0)}%, {(b.bbox[1] * 100).toFixed(0)}%
                                </td>
                                <td className="py-2 px-2.5 font-mono-data text-slate-600">
                                  {((b.bbox[2] - b.bbox[0]) * 100).toFixed(0)}×{((b.bbox[3] - b.bbox[1]) * 100).toFixed(0)}%
                                </td>
                                <td className="py-2 px-2.5 font-mono-data text-slate-500">{b.text.length}</td>
                                <td className="py-2 px-2.5 text-right">
                                  <button
                                    type="button"
                                    onClick={(e) => { e.stopPropagation(); copy(key, b.text); }}
                                    className="inline-flex items-center p-1 bg-slate-100 hover:bg-primary hover:text-white rounded cursor-pointer"
                                    title="Sao chép dòng này"
                                    aria-label="Sao chép dòng này"
                                  >
                                    {copiedKey === key ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
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

                {activeTab === 'text' && (
                  <div className="p-3.5 space-y-3 text-xs leading-relaxed max-h-[260px] overflow-y-auto custom-scrollbar font-body">
                    <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-[11px] text-slate-700 font-heading">
                          Văn bản theo thứ tự đọc ({activeBlocks.length} dòng)
                        </span>
                        <button
                          type="button"
                          onClick={() => copy('scene', buildSceneText(activeBlocks))}
                          disabled={activeBlocks.length === 0}
                          className="inline-flex items-center gap-1 px-2 py-0.5 bg-white hover:bg-primary hover:text-white border border-slate-200 rounded text-[10px] font-medium cursor-pointer disabled:opacity-40"
                        >
                          {copiedKey === 'scene' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                          Sao chép
                        </button>
                      </div>
                      {activeBlocks.length === 0 ? (
                        <p className="text-slate-400">Không có chữ đạt ngưỡng.</p>
                      ) : (
                        <p className="text-slate-800 whitespace-pre-wrap">
                          <Highlighted text={buildSceneText(activeBlocks)} query={search} />
                        </p>
                      )}
                      {stats.hidden > 0 && (
                        <p className="text-[10px] text-amber-600">{stats.hidden} dòng bị ẩn do ngưỡng độ tin cậy.</p>
                      )}
                    </div>
                    <details className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                      <summary className="cursor-pointer font-semibold text-[11px] text-slate-700 font-heading">
                        Chuỗi OCR thô lưu cho tóm tắt (ocr_text)
                      </summary>
                      <p className="mt-1.5 text-slate-700 whitespace-pre-wrap break-words">
                        {activeScene.ocrText || 'Trống.'}
                      </p>
                    </details>
                  </div>
                )}

                {activeTab === 'stats' && (
                  <div className="p-3.5 space-y-4 text-xs max-h-[260px] overflow-y-auto custom-scrollbar font-body">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {[
                        ['Dòng hiển thị', `${stats.kept}`],
                        ['Dòng bị ẩn', `${stats.hidden}`],
                        ['Ký tự', `${stats.chars}`],
                        ['Từ', `${stats.words}`],
                        ['Tin cậy TB', stats.kept ? `${(stats.avg * 100).toFixed(1)}%` : '—'],
                        ['Thấp nhất', stats.kept ? `${(stats.min * 100).toFixed(1)}%` : '—'],
                        ['Cao nhất', stats.kept ? `${(stats.max * 100).toFixed(1)}%` : '—'],
                        ['Diện tích chữ', `${(stats.coverage * 100).toFixed(1)}%`],
                      ].map(([k, v]) => (
                        <div key={k} className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                          <div className="text-[10px] text-slate-500">{k}</div>
                          <div className="font-semibold text-slate-900 font-mono-data">{v}</div>
                        </div>
                      ))}
                    </div>
                    <div>
                      <div className="text-[11px] font-semibold text-slate-700 mb-1.5 font-heading">Phân bố độ tin cậy (%)</div>
                      <div className="space-y-1">
                        {stats.histogram.map((n, i) => {
                          const max = Math.max(1, ...stats.histogram);
                          return (
                            <div key={i} className="flex items-center gap-2">
                              <span className="w-12 text-[10px] text-slate-500 font-mono-data">{HIST_LABELS[i]}</span>
                              <div className="flex-1 h-3 bg-slate-100 rounded overflow-hidden">
                                <div
                                  className={`h-full ${i >= 4 ? 'bg-emerald-500' : i >= 2 ? 'bg-amber-500' : 'bg-rose-500'}`}
                                  style={{ width: `${(n / max) * 100}%` }}
                                />
                              </div>
                              <span className="w-6 text-right text-[10px] font-mono-data text-slate-700">{n}</span>
                            </div>
                          );
                        })}
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1.5">
                        PaddleOCR loại sẵn các dòng dưới 50%, nên không có dữ liệu dưới ngưỡng này.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
