import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp?: string;
  avatarIcon: string;
  referenceTime?: number; // Time in seconds to seek to
}

export const QaPage: React.FC = () => {
  const toast = useToast();
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id') || '';

  const [videoData, setVideoData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      avatarIcon: 'auto_awesome',
      text: 'Chào bạn! Tôi có thể giúp bạn giải đáp thắc mắc gì về video bài giảng này không? Hãy chọn video bài giảng bên trái và nhập câu hỏi của bạn bên dưới.',
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

    const isValidUuid = (id: string) => {
      return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
    };

    if (videoId && isValidUuid(videoId)) {
      loadVideoAndScenes(videoId);
    } else {
      api.getVideos()
        .then(res => {
          if (res.success && res.data && res.data.length > 0) {
            const firstVideo = res.data[0];
            loadVideoAndScenes(firstVideo.videoId);
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
      sender: 'user',
      avatarIcon: 'person',
      text: queryText,
      timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMsg]);
    setInputText('');
    setIsTyping(true);

    if (!activeVideoId) {
      toast.error("Vui lòng chọn hoặc tải lên một video trước khi thực hiện đặt câu hỏi.", "Chưa chọn video");
      setIsTyping(false);
      return;
    }

    // Dynamic mock response generation
    setTimeout(() => {
      let responseText = "";
      let refTime: number | undefined = undefined;
      const lowerQuery = queryText.toLowerCase();

      if (lowerQuery.includes("tóm tắt") || lowerQuery.includes("summary") || lowerQuery.includes("nội dung")) {
        responseText = "Bài giảng này thảo luận về phương pháp phân tích bài giảng đa phương tiện. Nội dung chính bao gồm giới thiệu mục tiêu bài giảng, phân tích chi tiết dữ liệu hình ảnh/âm thanh và tóm tắt các điểm then chốt ở cuối chương. Bạn có thể xem chi tiết ở các mốc thời gian 01:25 và 05:12.";
        refTime = 85; // 01:25
      } else if (lowerQuery.includes("whisper") || lowerQuery.includes("clip") || lowerQuery.includes("gemini") || lowerQuery.includes("ai") || lowerQuery.includes("mô hình")) {
        responseText = "Hệ thống sử dụng mô hình WhisperX để bóc băng tiếng nói tiếng Việt chính xác, kết hợp CLIP để trích xuất đặc trưng hình ảnh của slide, và dùng mô hình ngôn ngữ lớn Gemini 1.5 để xâu chuỗi thông tin và trả lời câu hỏi.";
      } else if (lowerQuery.includes("slide") || lowerQuery.includes("hình ảnh") || lowerQuery.includes("khung hình")) {
        responseText = "Các slide bài giảng đã được tự động cắt lớp và gán thẻ thời gian dựa trên sự thay đổi khung hình chính (Keyframes). Bạn có thể nhấp trực tiếp vào danh sách 'Analyzed Segments' bên trái để tua nhanh đến slide tương ứng.";
      } else if (lowerQuery.includes("thời gian") || lowerQuery.includes("mốc") || lowerQuery.includes("khi nào") || lowerQuery.includes("bao lâu")) {
        responseText = "Mốc thời gian quan trọng của nội dung này nằm ở khoảng 02:40 trong video bài giảng. Hãy bấm vào nút tua nhanh bên cạnh câu trả lời này để di chuyển đến đúng phân đoạn đó.";
        refTime = 160; // 02:40
      } else {
        responseText = `Cám ơn câu hỏi của bạn về chủ đề "${queryText}". Dựa trên tài liệu bóc băng bài giảng, giảng viên giải thích chi tiết rằng cơ chế cốt lõi hoạt động dựa trên các tham số cấu hình hệ thống. Bạn có thể tua đến mốc 03:15 để nghe kỹ hơn phần này.`;
        refTime = 195; // 03:15
      }

      setMessages(prev => [...prev, {
        sender: 'bot',
        avatarIcon: 'auto_awesome',
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
        sender: 'bot',
        avatarIcon: 'auto_awesome',
        text: 'Lịch sử trò chuyện đã được xóa. Tôi sẵn sàng trả lời các câu hỏi mới!',
        timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    toast.success('Lịch sử trò chuyện đã được xóa.', 'Đã xóa');
  };

  return (
    <div className="fixed top-16 bottom-0 left-0 right-0 flex flex-col md:flex-row overflow-hidden bg-background text-on-surface">
      {/* Left Sidebar: Video & Key Segments */}
      <section className="w-full md:w-80 lg:w-96 bg-surface border-r border-outline-variant flex flex-col h-[40vh] md:h-full shrink-0 overflow-hidden">
        {/* Video Player Container */}
        <div className="p-4 border-b border-outline-variant/30">
          <div className="relative aspect-video bg-video-background rounded-xl overflow-hidden border border-outline-variant shadow-sm">
            {videoData?.filePath ? (
              <video 
                ref={videoRef}
                src={videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`} 
                poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
                controls
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center bg-slate-100 text-secondary p-4 text-center">
                <span className="material-symbols-outlined text-4xl mb-2 text-slate-400">videocam_off</span>
                <p className="text-xs">Chưa có video được tải hoặc chọn lọc.</p>
              </div>
            )}
          </div>
          <div className="mt-4">
            <h2 className="font-headline-md text-base font-bold text-deep-navy">
              {videoData?.title || 'Chưa chọn bài giảng'}
            </h2>
            <p className="font-body-sm text-xs text-secondary mt-1">
              Thời lượng: {videoData?.duration ? formatTimeText(videoData.duration) : '00:00'} • AI RAG Engine v2.4.0
            </p>
          </div>
        </div>

        {/* Key Segments List */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          <div className="flex items-center justify-between py-2 border-b border-outline-variant mb-4">
            <span className="font-label-md text-xs font-bold uppercase tracking-wider text-outline">Analyzed Segments</span>
            <span className="material-symbols-outlined text-outline text-lg">filter_list</span>
          </div>
          <div className="space-y-3">
            {loading ? (
              Array.from({ length: 3 }).map((_, idx) => (
                <div key={idx} className="p-3 rounded-lg border border-outline-variant space-y-2">
                  <Skeleton className="h-3 w-16 rounded" />
                  <Skeleton className="h-4 w-3/4 rounded" />
                  <Skeleton className="h-3 w-full rounded" />
                </div>
              ))
            ) : scenesList.length > 0 ? (
              scenesList.map((scene, idx) => (
                <div 
                  key={idx}
                  onClick={() => seekMiniPlayer(scene.startSeconds)}
                  className="p-3 rounded-lg border border-outline-variant hover:border-vibrant-cyan transition-colors cursor-pointer group"
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="bg-surface-container-highest px-2 py-0.5 rounded text-[10px] font-mono-data text-deep-navy">
                      {formatTimeText(scene.startSeconds)} - {formatTimeText(scene.endSeconds)}
                    </span>
                    <span className="material-symbols-outlined text-vibrant-cyan text-sm opacity-0 group-hover:opacity-100">shortcut</span>
                  </div>
                  <h4 className="font-label-md text-xs font-bold text-deep-navy mb-1">
                    Phân đoạn #{scene.sceneIndex !== undefined ? scene.sceneIndex + 1 : idx + 1}
                  </h4>
                  <p className="text-[11px] text-secondary line-clamp-2">
                    {scene.caption || scene.script || 'Đang xử lý phân đoạn...'}
                  </p>
                </div>
              ))
            ) : (
              <div className="p-4 text-center text-secondary border border-dashed border-outline-variant rounded-xl">
                <span className="material-symbols-outlined text-xl text-slate-400">segment</span>
                <p className="text-[10px] mt-1">Không có phân đoạn nào trong video này.</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Chat Interface */}
      <section className="flex-1 flex flex-col relative h-[60vh] md:h-full overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 z-0 opacity-[0.02] pointer-events-none overflow-hidden">
          <svg height="100%" width="100%" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern height="40" id="grid" patternUnits="userSpaceOnUse" width="40">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1"></path>
              </pattern>
            </defs>
            <rect fill="url(#grid)" height="100%" width="100%"></rect>
          </svg>
        </div>

        {/* Chat Header */}
        <div className="z-10 flex items-center justify-between px-6 py-4 glass-panel border-b border-outline-variant shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-deep-navy flex items-center justify-center text-white">
              <span className="material-symbols-outlined text-sm">auto_awesome</span>
            </div>
            <div>
              <h3 className="font-label-md text-sm font-bold text-deep-navy">Hỏi đáp Ngữ cảnh Video</h3>
              <span className="text-[10px] text-status-success uppercase font-bold tracking-tighter">AI Engine Online</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={clearHistory}
              className="px-3 py-1.5 hover:bg-surface-container-high rounded-lg text-secondary hover:text-primary transition-colors flex items-center gap-1.5 text-xs font-semibold"
              title="Xóa lịch sử"
            >
              <span className="material-symbols-outlined text-sm">delete_outline</span>
              <span className="hidden sm:inline">Xóa lịch sử</span>
            </button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6 z-10">
          {messages.map((msg, idx) => (
            <div 
              key={idx} 
              className={`flex flex-col max-w-[85%] ${msg.sender === 'user' ? 'items-end ml-auto' : 'items-start'}`}
            >
              <div className="flex items-center gap-2 mb-1 px-2">
                <span className={`font-label-sm text-xs font-semibold ${msg.sender === 'user' ? 'text-secondary' : 'text-deep-navy font-bold'}`}>
                  {msg.sender === 'user' ? 'Researcher' : 'Analysis Engine'}
                </span>
                {msg.sender === 'bot' && (
                  <span className="bg-vibrant-cyan/10 text-vibrant-cyan text-[8px] font-bold px-1.5 py-0.5 rounded uppercase">Verified</span>
                )}
              </div>
              <div 
                className={`p-4 text-sm ${
                  msg.sender === 'user' 
                    ? 'glass-chat-user rounded-2xl rounded-tr-none text-deep-navy' 
                    : 'glass-chat-ai rounded-2xl rounded-tl-none text-deep-navy shadow-sm'
                }`}
                dangerouslySetInnerHTML={{ __html: msg.text }}
              />
              {msg.referenceTime !== undefined && (
                <button 
                  onClick={() => seekMiniPlayer(msg.referenceTime!)}
                  className="flex items-center gap-2 bg-white/60 hover:bg-vibrant-cyan hover:text-white transition-all border border-vibrant-cyan/30 px-3 py-1.5 rounded-full mt-2 group shadow-sm text-xs"
                >
                  <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>play_circle</span>
                  <span className="font-mono-data font-bold">{formatTimeText(msg.referenceTime)}</span>
                  <span className="opacity-60 group-hover:opacity-100">Xem đoạn liên quan</span>
                </button>
              )}
              {msg.timestamp && (
                <span className={`text-[9px] text-outline mt-1 ${msg.sender === 'user' ? 'mr-2' : 'ml-2'}`}>
                  {msg.timestamp}
                </span>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="flex flex-col items-start">
              <div className="flex items-center gap-2 mb-1 px-2">
                <span className="font-label-sm text-xs text-deep-navy font-bold">Analysis Engine</span>
              </div>
              <div className="glass-chat-ai p-4 rounded-2xl rounded-tl-none flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-vibrant-cyan rounded-full animate-bounce"></div>
                <div className="w-1.5 h-1.5 bg-vibrant-cyan rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-1.5 h-1.5 bg-vibrant-cyan rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <span className="text-xs text-secondary italic ml-2">Đang phân tích...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input Area */}
        <div className="z-10 p-6 glass-panel border-t border-outline-variant shrink-0 bg-surface/50">
          <div className="relative max-w-4xl mx-auto">
            <div className="flex items-end gap-3 glass-panel p-2 rounded-[24px] border border-outline focus-within:border-vibrant-cyan focus-within:ring-1 focus-within:ring-vibrant-cyan transition-all shadow-lg bg-white">
              <button className="p-2 text-secondary hover:text-vibrant-cyan transition-colors ml-2 mb-1">
                <span className="material-symbols-outlined">attach_file</span>
              </button>
              <textarea 
                className="flex-1 bg-transparent border-none focus:ring-0 text-sm py-3 px-2 resize-none max-h-32 min-h-[48px] placeholder-outline-variant outline-none" 
                placeholder="Đặt câu hỏi về nội dung video bài giảng (Enter để gửi)..."
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
              <div className="flex items-center gap-1 mr-2 mb-1">
                <button className="p-2 text-secondary hover:text-vibrant-cyan transition-colors">
                  <span className="material-symbols-outlined">mic</span>
                </button>
                <button 
                  onClick={() => handleSendMessage()}
                  className="bg-deep-navy text-white w-10 h-10 rounded-full flex items-center justify-center hover:bg-vibrant-cyan transition-all shadow-md active:scale-95 shrink-0"
                >
                  <span className="material-symbols-outlined">send</span>
                </button>
              </div>
            </div>

            {/* Suggested Questions */}
            <div className="flex flex-wrap justify-center gap-4 mt-3">
              <button 
                onClick={() => handleSendMessage('Giải thích cơ chế Backpropagation trong Deep Learning?')}
                className="text-[10px] font-label-md text-secondary hover:text-vibrant-cyan uppercase tracking-widest flex items-center gap-1 font-semibold"
              >
                <span className="material-symbols-outlined text-[14px]">auto_awesome</span> Giải thích khái niệm chính
              </button>
              <button 
                onClick={() => handleSendMessage('Tóm tắt các tham số hiệu năng được đề cập trong bài giảng?')}
                className="text-[10px] font-label-md text-secondary hover:text-vibrant-cyan uppercase tracking-widest flex items-center gap-1 font-semibold"
              >
                <span className="material-symbols-outlined text-[14px]">table_chart</span> Thống kê số liệu
              </button>
              <button 
                onClick={() => handleSendMessage('So sánh Transformer và RNN theo phân tích của giảng viên?')}
                className="text-[10px] font-label-md text-secondary hover:text-vibrant-cyan uppercase tracking-widest flex items-center gap-1 font-semibold"
              >
                <span className="material-symbols-outlined text-[14px]">psychology</span> Phân tích logic bài giảng
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
