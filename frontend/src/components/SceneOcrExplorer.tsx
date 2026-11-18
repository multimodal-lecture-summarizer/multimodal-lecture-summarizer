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
import { useTranslation, Trans } from 'react-i18next';
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
  const { t } = useTranslation();
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
        throw new Error(res.message || t('scene.load_failed'));
      }
    } catch (err: any) {
      setError(err.message || t('scene.connection_error_ocr'));
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
            {t('scene.retry')}
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
          <h2 className="text-base font-bold text-slate-900 font-heading">{t('scene.ocr_title')}</h2>
          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700 border border-slate-200">PaddleOCR</span>
        </div>
        <p className="text-xs text-slate-500 mt-0.5 font-body">
          {t('scene.ocr_subtitle')}
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
          <p className="text-xs font-semibold text-slate-700 font-body">{t('scene.ocr_empty')}</p>
          <p className="text-[11px] text-slate-500 max-w-lg mx-auto font-body">
            {t('scene.ocr_empty_hint')}
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
          {pill(<Layers className="w-3.5 h-3.5 text-primary" />, t('scene.ocr_scenes_with_text'), `${totals.withText}/${scenes.length}`)}
          {pill(<ScanText className="w-3.5 h-3.5 text-slate-500" />, t('scene.ocr_lines'), String(totals.blocks))}
          {pill(<FileText className="w-3.5 h-3.5 text-slate-500" />, t('scene.ocr_words'), String(totals.words))}
          {pill(<BarChart3 className="w-3.5 h-3.5 text-slate-500" />, t('scene.ocr_avg_conf'), totals.blocks ? `${Math.round(totals.avg * 100)}%` : '—')}
          {hasAnyBlocks && (
            <button
              type="button"
              onClick={exportJson}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 font-medium cursor-pointer"
              title={t('scene.ocr_download')}
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
            placeholder={t('scene.ocr_search')}
            aria-label={t('scene.ocr_search_aria')}
            className="pl-8 pr-3 py-1.5 w-64 border border-slate-200 rounded-full text-xs focus:outline-none focus:ring-2 focus:ring-primary/30"
          />
        </div>
        {query && (
          <span className="text-slate-500">{t('scene.n_scenes_match', { n: filteredScenes.length })}</span>
        )}
        <span className="mx-1 text-slate-300">|</span>
        {chip('all', `${t('scene.all')} (${scenes.length})`)}
        {chip('with', t('scene.with_text'))}
        {chip('without', t('scene.without_text'))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Scene list */}
        <div className="lg:col-span-4 min-w-0 border border-slate-200 rounded-xl overflow-hidden bg-slate-50 flex flex-col max-h-[580px]">
          <div className="px-4 py-2.5 bg-white border-b border-slate-200 flex items-center justify-between">
            <span className="font-bold text-xs text-slate-700 flex items-center gap-1.5 font-heading">
              <Clock className="w-3.5 h-3.5 text-slate-400" /> {t('scene.lecture_scenes')}
            </span>
            <span className="text-[11px] text-slate-500 font-medium">{t('scene.n_scenes', { n: filteredScenes.length })}</span>
          </div>
          <div className="overflow-y-auto custom-scrollbar flex-1 p-2 space-y-1.5">
            {filteredScenes.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400 font-body">{t('scene.no_scene_match')}</div>
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
                          <img src={thumb} alt={t('scene.scene_n', { n: i + 1 })} className="w-full h-full object-cover" loading="lazy" />
                        ) : null}
                        <span className="absolute bottom-0.5 right-0.5 px-1 bg-black/80 text-[8px] font-mono-data font-bold text-white rounded">
                          {formatSeconds(s.startSec)}
                        </span>
                      </div>
                      <div className="flex-1 min-w-0 font-body">
                        <div className="flex items-center justify-between">
                          <span className={`text-xs font-bold ${isActive ? 'text-primary' : 'text-slate-900'}`}>
                            {t('scene.scene_n', { n: s.index })}
                          </span>
                          <span className="text-[10px] font-mono-data text-slate-500">
                            {formatSeconds(s.startSec)} - {formatSeconds(s.endSec)}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 mt-1">
                          {vis.length === 0 ? (
                            <span className="text-[9px] text-slate-400 italic">{t('scene.without_text')}</span>
                          ) : (
                            <>
                              <span className="px-1.5 rounded text-[9px] font-medium border bg-slate-50 text-slate-700 border-slate-200">
                                {t('scene.n_lines', { n: vis.length })}
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
            <div className="p-8 text-center text-xs text-slate-400 border border-slate-200 rounded-xl">{t('scene.pick_scene')}</div>
          ) : (
            <>
              <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
                <div className="px-4 py-2.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs text-white font-body">
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm font-heading">{t('scene.scene_n', { n: activeScene.index })}</span>
                      <span className="text-slate-500">|</span>
                      <span className="font-mono-data text-slate-300">
                        {formatSeconds(activeScene.startSec)} - {formatSeconds(activeScene.endSec)}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 flex items-center gap-1.5 flex-wrap">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <span><Trans i18nKey="scene.showing_lines" values={{ n: activeBlocks.length, pct: Math.round(minConf * 100) }} components={{ strong: <strong /> }} /></span>
                      {stats.hidden > 0 && (
                        <span className="text-amber-400 text-[10px]">· {t('scene.hidden_by_threshold', { n: stats.hidden })}</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3 flex-wrap">
                    <label className="flex items-center gap-1.5">
                      <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                      <span className="text-[10px] text-slate-400">{t('scene.threshold')}</span>
                      <input
                        type="range"
                        min={0.5}
                        max={0.95}
                        step={0.05}
                        value={minConf}
                        onChange={(e) => setMinConf(parseFloat(e.target.value))}
                        className="w-14 h-1 bg-slate-700 rounded cursor-pointer accent-primary"
                        aria-label={t('scene.ocr_threshold_aria')}
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
                      {showBoxes ? t('scene.show_boxes') : t('scene.hide_boxes')}
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowLabels(!showLabels)}
                      disabled={!showBoxes}
                      className={`px-2 py-1 rounded text-xs font-medium cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                        showLabels ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                      }`}
                    >
                      {t('scene.text_labels')}
                    </button>
                    <button
                      type="button"
                      onClick={() => onSeekVideo(activeScene.startSec)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 bg-primary hover:bg-primary-hover text-white rounded text-xs font-medium cursor-pointer shadow-xs"
                    >
                      <Play className="w-3 h-3 fill-current" /> {t('scene.seek_video')}
                    </button>
                  </div>
                </div>

                {/* Canvas: wrapper takes the image's own size so normalized boxes line up with pixels */}
                <div className="w-full bg-slate-950 flex items-center justify-center overflow-hidden select-none p-1">
                  {imageUrl ? (
                    <div className="relative inline-block max-w-full">
                      <img src={imageUrl} alt={t('scene.keyframe_alt', { n: activeScene.index })} className="block max-w-full max-h-[360px] w-auto h-auto" />
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
                    <div className="text-slate-500 text-xs font-body py-16">{t('scene.no_thumb')}</div>
                  )}
                </div>

                {/* Legend */}
                <div className="px-4 py-1.5 bg-slate-900 border-t border-slate-800 flex items-center gap-4 text-[10px] text-slate-400">
                  <span>{t('scene.confidence_label')}</span>
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
                  {tabBtn('blocks', <ListFilter className="w-3.5 h-3.5" />, t('scene.text_blocks', { n: activeBlocks.length }))}
                  {tabBtn('text', <FileText className="w-3.5 h-3.5" />, t('scene.full_text'))}
                  {tabBtn('stats', <BarChart3 className="w-3.5 h-3.5" />, t('scene.stats'))}
                </div>

                {activeTab === 'blocks' && (
                  <div className="p-3 max-h-[260px] overflow-y-auto overflow-x-auto custom-scrollbar">
                    {activeBlocks.length === 0 ? (
                      <div className="p-6 text-center text-xs text-slate-400 font-body space-y-1">
                        <p>{t('scene.no_text_above', { pct: Math.round(minConf * 100) })}</p>
                        {stats.hidden > 0 && (
                          <p className="text-[11px] text-amber-500 font-medium">{t('scene.lines_below_hint', { n: stats.hidden })}</p>
                        )}
                      </div>
                    ) : (
                      <table className="w-full text-xs text-left font-body">
                        <thead>
                          <tr className="border-b border-slate-200 text-slate-500 text-[10px] font-semibold uppercase tracking-wider sticky top-0 bg-white z-10">
                            <th className="py-1.5 px-2.5">#</th>
                            <th className="py-1.5 px-2.5">{t('scene.col_text')}</th>
                            <th className="py-1.5 px-2.5">{t('scene.col_confidence')}</th>
                            <th className="py-1.5 px-2.5">{t('scene.col_position')}</th>
                            <th className="py-1.5 px-2.5">{t('scene.col_size')}</th>
                            <th className="py-1.5 px-2.5">{t('scene.col_chars')}</th>
                            <th className="py-1.5 px-2.5 text-right">{t('scene.col_copy')}</th>
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
                                    title={t('scene.copy_line')}
                                    aria-label={t('scene.copy_line')}
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
                          {t('scene.reading_order', { n: activeBlocks.length })}
                        </span>
                        <button
                          type="button"
                          onClick={() => copy('scene', buildSceneText(activeBlocks))}
                          disabled={activeBlocks.length === 0}
                          className="inline-flex items-center gap-1 px-2 py-0.5 bg-white hover:bg-primary hover:text-white border border-slate-200 rounded text-[10px] font-medium cursor-pointer disabled:opacity-40"
                        >
                          {copiedKey === 'scene' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                          {t('scene.copy')}
                        </button>
                      </div>
                      {activeBlocks.length === 0 ? (
                        <p className="text-slate-400">{t('scene.no_text_threshold')}</p>
                      ) : (
                        <p className="text-slate-800 whitespace-pre-wrap">
                          <Highlighted text={buildSceneText(activeBlocks)} query={search} />
                        </p>
                      )}
                      {stats.hidden > 0 && (
                        <p className="text-[10px] text-amber-600">{t('scene.hidden_by_conf', { n: stats.hidden })}</p>
                      )}
                    </div>
                    <details className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                      <summary className="cursor-pointer font-semibold text-[11px] text-slate-700 font-heading">
                        {t('scene.raw_ocr')}
                      </summary>
                      <p className="mt-1.5 text-slate-700 whitespace-pre-wrap break-words">
                        {activeScene.ocrText || t('scene.empty')}
                      </p>
                    </details>
                  </div>
                )}

                {activeTab === 'stats' && (
                  <div className="p-3.5 space-y-4 text-xs max-h-[260px] overflow-y-auto custom-scrollbar font-body">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {[
                        [t('scene.st_shown'), `${stats.kept}`],
                        [t('scene.st_hidden'), `${stats.hidden}`],
                        [t('scene.col_chars'), `${stats.chars}`],
                        [t('scene.st_words'), `${stats.words}`],
                        [t('scene.st_avg'), stats.kept ? `${(stats.avg * 100).toFixed(1)}%` : '—'],
                        [t('scene.st_min'), stats.kept ? `${(stats.min * 100).toFixed(1)}%` : '—'],
                        [t('scene.st_max'), stats.kept ? `${(stats.max * 100).toFixed(1)}%` : '—'],
                        [t('scene.st_coverage'), `${(stats.coverage * 100).toFixed(1)}%`],
                      ].map(([k, v]) => (
                        <div key={k} className="p-2 bg-slate-50 border border-slate-200 rounded-lg">
                          <div className="text-[10px] text-slate-500">{k}</div>
                          <div className="font-semibold text-slate-900 font-mono-data">{v}</div>
                        </div>
                      ))}
                    </div>
                    <div>
                      <div className="text-[11px] font-semibold text-slate-700 mb-1.5 font-heading">{t('scene.conf_distribution')}</div>
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
                        {t('scene.paddle_note')}
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
