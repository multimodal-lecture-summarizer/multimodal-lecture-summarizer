import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, VideoOff, Trash2, Clock as HistoryIcon, MessageSquare,
  Send, Sparkles
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
      api.getVideo(id)
      .then((videoRes) => {
        if (videoRes.success && videoRes.data) {
          setVideoData(videoRes.data);
          setActiveVideoId(videoRes.data.videoId);
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
    if (!activeVideoId) return;

    api.getQaHistory(activeVideoId)
      .then(res => {
        if (res.success && res.data && res.data.length > 0) {
          const loadedMsgs: Message[] = [];
          res.data.forEach((log: any) => {
            const rawDate = log.askedAt || log.createdAt;
            const rawDateStr = typeof rawDate === 'string' && !rawDate.endsWith('Z') && !rawDate.includes('+') ? `${rawDate}Z` : rawDate;
            const timeStr = rawDate 
              ? new Date(rawDateStr).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }) 
              : '';
            
            loadedMsgs.push({
              id: `user-${log.qaId}`,
              sender: 'user',
              text: log.question,
              timestamp: timeStr
            });

            loadedMsgs.push({
              id: `bot-${log.qaId}`,
              sender: 'bot',
              text: log.answer,
              referenceTime: log.referenceTime,
              timestamp: timeStr
            });
          });
          setMessages(loadedMsgs);
        }
      })
      .catch(err => {
        console.error("Failed to load past QA chat history:", err);
      });
  }, [activeVideoId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);


  const handleSendMessage = async (textToSend = inputText) => {
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

    try {
      const res = await api.askQuestion(activeVideoId, queryText);
      if (res.success && res.data) {
        const qaData = res.data;
        const answerText = qaData.answer || '';
        const refTime = qaData.referenceTime ?? (qaData.citations && qaData.citations.length > 0 ? qaData.citations[0].startSeconds : undefined);

        setMessages(prev => [...prev, {
          id: `msg-${Date.now() + 1}`,
          sender: 'bot',
          text: answerText,
          referenceTime: refTime,
          timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
        }]);
      } else {
        throw new Error(res.message || 'Không thể lấy câu trả lời từ AI');
      }
    } catch (err: any) {
      console.error("QA error:", err);
      toast.error(err.message || 'Hỏi đáp AI gặp lỗi', 'Lỗi RAG Q&A');
      setMessages(prev => [...prev, {
        id: `msg-${Date.now() + 1}`,
        sender: 'bot',
        text: `RAG Q&A: ${err.message || 'Không thể kết nối tới mô hình AI'}. Vui lòng thử lại sau.`,
        timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const formatTimeText = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = Math.floor(secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const formatMessageHtml = (rawText: string) => {
    if (!rawText) return '';
    
    let text = rawText;

    // 1. Replace [MM:SS] or [MM:SS - MM:SS] or **[MM:SS]** timestamps with interactive seek buttons FIRST
    text = text.replace(/(\*\*|\*)?\[(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?)?\](\*\*|\*)?/g, (_match, _b1, h1, m1, s1, h2, m2, s2) => {
      let startSecs = 0;
      let label = '';
      if (s1 !== undefined) {
        startSecs = parseInt(h1, 10) * 3600 + parseInt(m1, 10) * 60 + parseInt(s1, 10);
        label = `${h1}:${m1}:${s1}`;
      } else {
        startSecs = parseInt(h1, 10) * 60 + parseInt(m1, 10);
        label = `${h1}:${m1}`;
      }

      if (h2 !== undefined) {
        if (s2 !== undefined) {
          label += ` - ${h2}:${m2}:${s2}`;
        } else {
          label += ` - ${h2}:${m2}`;
        }
      }

      return `<button type="button" data-seek="${startSecs}" class="inline-flex items-center gap-1 font-mono text-xs font-bold px-2 py-0.5 rounded-md bg-primary/15 text-primary hover:bg-primary hover:text-white transition-all shadow-sm mx-1 my-0.5 cursor-pointer border border-primary/20">▶ [${label}]</button>`;
    });

    // 2. Transform **bold** to <strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-slate-900">$1</strong>');
    
    // 3. Transform Headings
    text = text.replace(/^### (.*$)/gim, '<h4 class="font-bold text-slate-900 text-base mt-3 mb-1">$1</h4>');
    text = text.replace(/^## (.*$)/gim, '<h3 class="font-bold text-slate-900 text-lg mt-3 mb-1">$1</h3>');

    // 4. Fix standalone bullets on single lines (e.g. •\nText)
    text = text.replace(/([•\-\*])\s*\n\s*/g, '$1 ');

    // 5. Clean list items (- item, * item, • item)
    text = text.replace(/^[\s]*[•\-\*]\s*(.*$)/gim, '<div class="flex items-start gap-2 my-1.5 pl-1"><span class="text-primary font-bold inline-block mt-0.5">•</span><span class="flex-1">$1</span></div>');

    // 6. Line breaks handling
    text = text.replace(/\n{2,}/g, '<div class="h-2"></div>');
    text = text.replace(/\n/g, '<br />');

    return text;
  };

  const handleChatContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = (e.target as HTMLElement).closest('[data-seek]');
    if (target) {
      e.preventDefault();
      const seekSecs = parseFloat(target.getAttribute('data-seek') || '0');
      seekMiniPlayer(seekSecs);
    }
  };

  const clearHistory = async () => {
    if (activeVideoId) {
      try {
        await api.clearQaHistory(activeVideoId);
      } catch (err) {
        console.error("Failed to delete QA history on backend:", err);
      }
    }

    setMessages([
      {
        id: `msg-${Date.now()}`,
        sender: 'bot',
        text: t('qa.history_cleared_msg') || 'Lịch sử trò chuyện đã được xóa. Tôi sẵn sàng trả lời các câu hỏi mới!',
        timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    toast.success(t('qa.clear_success') || 'Lịch sử trò chuyện đã được xóa.', t('qa.clear_success_title') || 'Đã xóa');
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
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <HistoryIcon size={14} className="text-primary" />
              {t('qa.chat_history') || 'Lịch sử câu hỏi & Chat'}
            </span>
            {messages.filter(m => m.sender === 'user').length > 0 && (
              <span className="text-[10px] font-mono font-bold bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                {messages.filter(m => m.sender === 'user').length} câu hỏi
              </span>
            )}
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
            ) : messages.filter(m => m.sender === 'user').length > 0 ? (
              messages
                .map((msg, idx) => ({ msg, idx }))
                .filter(({ msg }) => msg.sender === 'user')
                .map(({ msg, idx }, qIdx) => {
                  const botMsg = messages[idx + 1] && messages[idx + 1].sender === 'bot' ? messages[idx + 1] : null;
                  const cleanBotText = botMsg ? botMsg.text.replace(/<[^>]*>?/gm, '').replace(/▶\s*\[\d{2}:\d{2}\]/g, '').trim() : '';

                  return (
                    <motion.div 
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      key={msg.id}
                      onClick={() => {
                        const el = document.getElementById(msg.id);
                        if (el) {
                          el.scrollIntoView({ behavior: 'smooth' });
                        }
                      }}
                      className="p-3.5 rounded-2xl border border-slate-200/60 shadow-sm hover:border-primary/50 hover:shadow-md cursor-pointer group bg-white transition-all"
                    >
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="bg-primary/10 text-primary px-2 py-0.5 rounded-md text-[10px] font-mono font-bold">
                          #Câu hỏi {qIdx + 1}
                        </span>
                        {msg.timestamp && (
                          <span className="text-[10px] font-mono text-slate-400">
                            {msg.timestamp}
                          </span>
                        )}
                      </div>
                      <h4 className="font-heading text-xs font-bold text-slate-900 group-hover:text-primary transition-colors line-clamp-2 leading-snug mb-1">
                        {msg.text}
                      </h4>
                      {cleanBotText && (
                        <p className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed bg-slate-50/80 p-2 rounded-xl border border-slate-100 mt-1.5">
                          {cleanBotText}
                        </p>
                      )}
                    </motion.div>
                  );
                })
            ) : (
              <div className="p-6 text-center text-slate-400 border-2 border-dashed border-slate-200/60 rounded-2xl">
                <MessageSquare size={24} className="mx-auto mb-2 text-slate-300" />
                <p className="text-xs font-semibold text-slate-600">Chưa có lịch sử câu hỏi nào</p>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">Hãy nhập câu hỏi ở ô bên phải để bắt đầu trò chuyện RAG với AI</p>
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
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6 relative" onClick={handleChatContainerClick}>
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div 
                key={msg.id}
                id={msg.id}
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
                  dangerouslySetInnerHTML={{ __html: formatMessageHtml(msg.text) }}
                />
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
            <div className="flex flex-wrap items-center justify-center gap-2 mb-4">
              <button 
                onClick={() => handleSendMessage(t('qa.suggest1_q'))} 
                className="px-3.5 py-1.5 bg-slate-100/80 hover:bg-primary/10 hover:text-primary text-slate-700 rounded-full text-xs font-medium border border-slate-200/60 shadow-sm transition-all flex items-center gap-1.5 cursor-pointer"
              >
                {t('qa.suggest1')}
              </button>
              <button 
                onClick={() => handleSendMessage(t('qa.suggest2_q'))} 
                className="px-3.5 py-1.5 bg-slate-100/80 hover:bg-primary/10 hover:text-primary text-slate-700 rounded-full text-xs font-medium border border-slate-200/60 shadow-sm transition-all flex items-center gap-1.5 cursor-pointer"
              >
                {t('qa.suggest2')}
              </button>
              <button 
                onClick={() => handleSendMessage(t('qa.suggest3_q'))} 
                className="px-3.5 py-1.5 bg-slate-100/80 hover:bg-primary/10 hover:text-primary text-slate-700 rounded-full text-xs font-medium border border-slate-200/60 shadow-sm transition-all flex items-center gap-1.5 cursor-pointer"
              >
                {t('qa.suggest3')}
              </button>
              <button 
                onClick={() => handleSendMessage(t('qa.suggest4_q'))} 
                className="px-3.5 py-1.5 bg-slate-100/80 hover:bg-primary/10 hover:text-primary text-slate-700 rounded-full text-xs font-medium border border-slate-200/60 shadow-sm transition-all flex items-center gap-1.5 cursor-pointer"
              >
                {t('qa.suggest4')}
              </button>
            </div>

            <div className="flex items-end gap-3 bg-white p-2 rounded-3xl border-2 border-slate-200 focus-within:border-primary focus-within:shadow-lg focus-within:shadow-primary/10 transition-all">
              <textarea 
                className="flex-1 bg-transparent border-none focus:ring-0 text-sm py-4 px-4 resize-none max-h-32 min-h-[52px] placeholder-slate-400 outline-none text-slate-700" 
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
              <div className="flex items-center mr-1 mb-1 shrink-0">
                <motion.button 
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleSendMessage()}
                  className="bg-primary text-white w-12 h-12 rounded-full flex items-center justify-center hover:bg-primary-hover shadow-md shadow-primary/30 transition-all cursor-pointer"
                >
                  <Send size={18} className="ml-0.5" />
                </motion.button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
