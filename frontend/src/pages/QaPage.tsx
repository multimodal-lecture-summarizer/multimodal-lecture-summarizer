import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, Play, VideoOff, Filter, Trash2, 
  Send, Mic, Paperclip, Sparkles, Table, BrainCircuit
} from 'lucide-react';

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp?: string;
  referenceTime?: number;
}

export const QaPage: React.FC = () => {
  const toast = useToast();
  const { t } = useTranslation();
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id') || '';

  const [videoData, setVideoData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'msg-1',
      sender: 'bot',
      text: t('qa.welcome_msg') || 'Hi there! I am your AI assistant. Ask me anything about the video.',
      timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [scenesList, setScenesList] = useState<any[]>([]);
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);

  const seekMiniPlayer = (secs: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = secs;
      videoRef.current.play().catch(() => {});
    }
  };

  useEffect(() => {
    setLoading(true);
    
    const loadVideoAndScenes = (id: string) => {
      Promise.all([
        api.getVideo(id),
        api.getVideoScenes(id).catch(() => ({ success: false, data: [] }))
      ])
      .then(([videoRes, scenesRes]) => {
        if (videoRes.success && videoRes.data) {
          setVideoData(videoRes.data);
          setActiveVideoId(videoRes.data.videoId);
        }
        if (scenesRes.success && scenesRes.data && scenesRes.data.length > 0) {
          setScenesList(scenesRes.data);
        } else {
          setScenesList([]);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load details for video:", id, err);
        setLoading(false);
        setActiveVideoId(null);
      });
    };

    const isValidUuid = (id: string) => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);

    if (videoId && isValidUuid(videoId)) {
      loadVideoAndScenes(videoId);
    } else {
      api.getVideos()
        .then(res => {
          if (res.success && res.data && res.data.length > 0) {
            loadVideoAndScenes(res.data[0].videoId);
          } else {
            setLoading(false);
            setActiveVideoId(null);
          }
        })
        .catch(err => {
          console.error("Failed to fetch videos list for QA default:", err);
          setLoading(false);
          setActiveVideoId(null);
        });
    }
  }, [videoId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSendMessage = (textToSend = inputText) => {
    if (!textToSend.trim()) return;

    const queryText = textToSend;
    const newMsg: Message = {
      id: `msg-${Date.now()}`,
      sender: 'user',
      text: queryText,
      timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMsg]);
    setInputText('');
    setIsTyping(true);

    if (!activeVideoId) {
      toast.error(t('qa.no_video_selected'), t('qa.no_video_alert'));
      setIsTyping(false);
      return;
    }

    setTimeout(() => {
      let responseText = "";
      let refTime: number | undefined = undefined;
      const lowerQuery = queryText.toLowerCase();

      if (lowerQuery.includes("tóm tắt") || lowerQuery.includes("summary")) {
        responseText = "Bài giảng này thảo luận về phương pháp phân tích bài giảng đa phương tiện. Nội dung chính bao gồm giới thiệu mục tiêu bài giảng, phân tích chi tiết dữ liệu hình ảnh/âm thanh và tóm tắt các điểm then chốt ở cuối chương. Bạn có thể xem chi tiết ở các mốc thời gian 01:25 và 05:12.";
        refTime = 85; 
      } else if (lowerQuery.includes("whisper") || lowerQuery.includes("clip") || lowerQuery.includes("ai")) {
        responseText = "Hệ thống sử dụng mô hình WhisperX để bóc băng tiếng nói tiếng Việt chính xác, kết hợp CLIP để trích xuất đặc trưng hình ảnh của slide, và dùng mô hình ngôn ngữ lớn Gemini 1.5 để xâu chuỗi thông tin và trả lời câu hỏi.";
      } else if (lowerQuery.includes("thời gian") || lowerQuery.includes("khi nào")) {
        responseText = "Mốc thời gian quan trọng của nội dung này nằm ở khoảng 02:40 trong video bài giảng. Hãy bấm vào nút tua nhanh bên cạnh câu trả lời này để di chuyển đến đúng phân đoạn đó.";
        refTime = 160; 
      } else {
        responseText = `Cám ơn câu hỏi của bạn về chủ đề "${queryText}". Dựa trên tài liệu bóc băng bài giảng, giảng viên giải thích chi tiết rằng cơ chế cốt lõi hoạt động dựa trên các tham số cấu hình hệ thống. Bạn có thể tua đến mốc 03:15 để nghe kỹ hơn phần này.`;
        refTime = 195;
      }

      setMessages(prev => [...prev, {
        id: `msg-${Date.now() + 1}`,
        sender: 'bot',
        text: responseText,
        referenceTime: refTime,
        timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
      }]);
      setIsTyping(false);
    }, 1200);
  };

  const formatTimeText = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = Math.floor(secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const clearHistory = () => {
    setMessages([
      {
        id: `msg-${Date.now()}`,
        sender: 'bot',
        text: t('qa.history_cleared_msg') || 'History cleared.',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    toast.success(t('qa.clear_success'), t('qa.clear_success_title'));
  };

  return (
    <div className="flex flex-col md:flex-row h-[calc(100vh-64px)] bg-[#FAF5FF] overflow-hidden">
      {/* Left Sidebar: Video & Key Segments */}
      <section className="w-full md:w-80 lg:w-[400px] bg-slate-100/60 backdrop-blur-xl border-r border-slate-200 flex flex-col h-[40vh] md:h-full shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.02)] z-10">
        <div className="p-4 lg:p-6 border-b border-slate-200/50">
          <div className="relative aspect-video bg-black rounded-2xl overflow-hidden shadow-lg border border-slate-200/50">
            {videoData?.filePath ? (
              <video 
                ref={videoRef}
                src={videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`} 
                poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
                controls
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center bg-slate-100 text-slate-500 p-4 text-center">
                <VideoOff size={32} className="mb-2 text-slate-400" />
                <p className="text-xs font-medium">{t('qa.no_video_title')}</p>
              </div>
            )}
          </div>
          <div className="mt-5 px-1">
            <h2 className="font-heading text-lg font-bold text-slate-900 leading-tight">
              {videoData?.title || t('qa.not_selected')}
            </h2>
            <div className="flex items-center gap-2 mt-2">
              <span className="font-mono text-xs font-semibold bg-slate-200/50 px-2 py-1 rounded-md text-slate-600">
                {videoData?.duration ? formatTimeText(videoData.duration) : '00:00'}
              </span>
              <span className="text-xs text-primary font-bold flex items-center gap-1 bg-primary/10 px-2 py-1 rounded-md">
                <Sparkles size={12} /> AI RAG v2.4
              </span>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 lg:p-6">
          <div className="flex items-center justify-between mb-4 px-1">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">{t('qa.analyzed_segments')}</span>
            <button className="text-slate-400 hover:text-primary transition-colors">
              <Filter size={16} />
            </button>
          </div>
          <div className="space-y-3">
            {loading ? (
              Array.from({ length: 3 }).map((_, idx) => (
                <div key={idx} className="p-4 rounded-2xl border border-slate-200/50 space-y-3 bg-white/50">
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-full" />
                </div>
              ))
            ) : scenesList.length > 0 ? (
              scenesList.map((scene, idx) => (
                <motion.div 
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  key={idx}
                  onClick={() => seekMiniPlayer(scene.startSeconds)}
                  className="p-4 rounded-2xl border border-slate-200/60 shadow-sm hover:border-primary/50 hover:shadow-md cursor-pointer group bg-white transition-all"
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="bg-slate-100 border border-slate-200/50 px-2 py-1 rounded-lg text-[10px] font-mono font-bold text-slate-700">
                      {formatTimeText(scene.startSeconds)} - {formatTimeText(scene.endSeconds)}
                    </span>
                    <Play size={14} className="text-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                  <h4 className="font-heading text-sm font-bold text-slate-900 mb-1">
                    {t('qa.segment')} #{scene.sceneIndex !== undefined ? scene.sceneIndex + 1 : idx + 1}
                  </h4>
                  <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                    {scene.caption || scene.script || t('qa.processing_segment')}
                  </p>
                </motion.div>
              ))
            ) : (
              <div className="p-6 text-center text-slate-400 border-2 border-dashed border-slate-200/60 rounded-2xl">
                <Sparkles size={24} className="mx-auto mb-2 opacity-50" />
                <p className="text-xs font-medium">{t('qa.no_segments')}</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Chat Interface */}
      <section className="flex-1 flex flex-col relative h-[60vh] md:h-full bg-white">
        
        {/* Chat Header */}
        <div className="z-20 flex items-center justify-between px-6 py-4 glass border-b border-slate-200/50 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-indigo-600 text-white flex items-center justify-center shadow-lg shadow-primary/20">
              <Bot size={20} />
            </div>
            <div>
              <h3 className="font-heading text-base font-bold text-slate-900">{t('qa.header_title') || 'AI Assistant'}</h3>
              <div className="flex items-center gap-1.5 text-[11px] text-emerald-500 font-bold uppercase tracking-wider">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                Online
              </div>
            </div>
          </div>
          <button 
            onClick={clearHistory}
            className="p-2 text-slate-400 hover:text-danger hover:bg-red-50 rounded-lg transition-colors flex items-center gap-2 text-sm font-semibold"
            title={t('qa.clear_history')}
          >
            <Trash2 size={18} />
            <span className="hidden sm:inline">{t('qa.clear_history')}</span>
          </button>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6 relative">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div 
                key={msg.id}
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className={`flex flex-col max-w-[85%] ${msg.sender === 'user' ? 'items-end ml-auto' : 'items-start'}`}
              >
                <div className="flex items-center gap-2 mb-1 px-1">
                  <span className={`text-[11px] font-bold uppercase tracking-widest ${msg.sender === 'user' ? 'text-slate-400' : 'text-primary'}`}>
                    {msg.sender === 'user' ? t('qa.role_user') : t('qa.role_ai')}
                  </span>
                </div>
                <div 
                  className={`p-4 text-sm leading-relaxed shadow-sm ${
                    msg.sender === 'user' 
                      ? 'bg-slate-900 text-white rounded-2xl rounded-tr-sm' 
                      : 'bg-slate-50 border border-slate-200/50 rounded-2xl rounded-tl-sm text-slate-800'
                  }`}
                  dangerouslySetInnerHTML={{ __html: msg.text }}
                />
                {msg.referenceTime !== undefined && (
                  <motion.button 
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => seekMiniPlayer(msg.referenceTime!)}
                    className="flex items-center gap-2 bg-white hover:bg-primary hover:text-white hover:border-primary transition-all border border-slate-200 shadow-sm px-4 py-1.5 rounded-full mt-2 group text-xs text-slate-700 font-semibold"
                  >
                    <Play size={14} className="text-primary group-hover:text-white" />
                    <span className="font-mono">{formatTimeText(msg.referenceTime)}</span>
                    <span className="opacity-0 group-hover:opacity-100 -ml-2 group-hover:ml-0 transition-all overflow-hidden whitespace-nowrap max-w-0 group-hover:max-w-[100px]">
                      {t('qa.view_segment')}
                    </span>
                  </motion.button>
                )}
                {msg.timestamp && (
                  <span className={`text-[10px] font-mono text-slate-400 mt-1.5 ${msg.sender === 'user' ? 'mr-1' : 'ml-1'}`}>
                    {msg.timestamp}
                  </span>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {isTyping && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-start">
              <div className="flex items-center gap-2 mb-1 px-1">
                <span className="text-[11px] font-bold uppercase tracking-widest text-primary">{t('qa.role_ai')}</span>
              </div>
              <div className="bg-slate-50 border border-slate-200/50 p-4 rounded-2xl rounded-tl-sm flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                </div>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input Area */}
        <div className="z-20 p-4 md:p-6 bg-white/80 backdrop-blur-md border-t border-slate-200/50 shrink-0">
          <div className="relative max-w-4xl mx-auto">
            {/* Suggested Questions */}
            <div className="flex flex-wrap items-center justify-center gap-3 mb-4">
              <button onClick={() => handleSendMessage('Giải thích cơ chế Backpropagation trong Deep Learning?')} className="px-3 py-1.5 bg-slate-100 hover:bg-primary/10 hover:text-primary text-slate-600 rounded-full text-[11px] font-semibold transition-colors flex items-center gap-1.5">
                <Sparkles size={12} /> {t('qa.suggest1')}
              </button>
              <button onClick={() => handleSendMessage('Tóm tắt các tham số hiệu năng được đề cập trong bài giảng?')} className="px-3 py-1.5 bg-slate-100 hover:bg-primary/10 hover:text-primary text-slate-600 rounded-full text-[11px] font-semibold transition-colors flex items-center gap-1.5">
                <Table size={12} /> {t('qa.suggest2')}
              </button>
              <button onClick={() => handleSendMessage('So sánh Transformer và RNN theo phân tích của giảng viên?')} className="px-3 py-1.5 bg-slate-100 hover:bg-primary/10 hover:text-primary text-slate-600 rounded-full text-[11px] font-semibold transition-colors flex items-center gap-1.5">
                <BrainCircuit size={12} /> {t('qa.suggest3')}
              </button>
            </div>

            <div className="flex items-end gap-3 bg-white p-2 rounded-3xl border-2 border-slate-200 focus-within:border-primary focus-within:shadow-lg focus-within:shadow-primary/10 transition-all">
              <button className="p-3 text-slate-400 hover:text-primary transition-colors rounded-full hover:bg-slate-50 mb-0.5 ml-1 shrink-0">
                <Paperclip size={20} />
              </button>
              <textarea 
                className="flex-1 bg-transparent border-none focus:ring-0 text-sm py-4 px-2 resize-none max-h-32 min-h-[52px] placeholder-slate-400 outline-none text-slate-700" 
                placeholder={t('qa.input_placeholder') || "Ask a question..."}
                rows={1}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
              <div className="flex items-center gap-2 mr-1 mb-1 shrink-0">
                <button className="p-3 text-slate-400 hover:text-primary transition-colors rounded-full hover:bg-slate-50 hidden sm:block">
                  <Mic size={20} />
                </button>
                <motion.button 
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleSendMessage()}
                  className="bg-primary text-white w-12 h-12 rounded-full flex items-center justify-center hover:bg-primary-hover shadow-md shadow-primary/30 transition-all"
                >
                  <Send size={18} className="ml-1" />
                </motion.button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
