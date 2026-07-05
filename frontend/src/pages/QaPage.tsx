import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { CONFIG } from '../config';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp?: string;
  avatarIcon: string;
  referenceTime?: number; // Time in seconds to seek to
}

export const QaPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id') || '';

  const [videoData, setVideoData] = useState<any>(null);
  const [, setLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      avatarIcon: 'auto_awesome',
      text: 'Chào bạn! Hệ thống RAG đã được nạp toàn bộ Transcript và thông tin Keyframes (BLIP-2) của video. Bạn có muốn hỏi gì về bài giảng này không?'
    },
    {
      sender: 'user',
      avatarIcon: 'person',
      text: 'Giảng viên định nghĩa thế nào về kiến trúc Transformer? Tại sao nó tốt hơn RNN?'
    },
    {
      sender: 'bot',
      avatarIcon: 'auto_awesome',
      text: 'Theo bài giảng, kiến trúc Transformer là một mô hình học sâu chủ yếu dựa trên cơ chế <strong>Self-Attention</strong>. Giảng viên giải thích rằng khác với RNN (Recurrent Neural Networks) phải xử lý dữ liệu tuần tự từng từ một, Transformer có khả năng xem xét toàn bộ câu cùng một lúc.<br/><br/>Điều này mang lại hai lợi ích chính:<ul class="mt-2.5 pl-5 list-disc"><li>Khắc phục triệt để vấn đề mất mát thông tin đối với chuỗi văn bản dài.</li><li>Cho phép tính toán song song trên GPU, giúp tốc độ huấn luyện nhanh hơn rất nhiều.</li></ul>',
      referenceTime: 252 // 04:12 in seconds
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const seekMiniPlayer = (secs: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = secs;
      videoRef.current.play().catch(() => {});
    }
  };

  useEffect(() => {
    if (!videoId) return;
    setLoading(true);
    api.getVideo(videoId)
      .then(res => {
        if (res.success && res.data) {
          setVideoData(res.data);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load video details for QA:", err);
        setLoading(false);
      });
  }, [videoId]);

  const handleSendMessage = (textToSend = inputText) => {
    if (!textToSend.trim()) return;

    const queryText = textToSend;
    const newMsg: Message = {
      sender: 'user',
      avatarIcon: 'person',
      text: queryText
    };

    setMessages(prev => [...prev, newMsg]);
    setInputText('');
    setIsTyping(true);

    api.askQuestion(videoId, queryText)
      .then(res => {
        if (res.success && res.data) {
          let refTime: number | undefined = undefined;
          
          // Parse timestamp from answer like "05:12" or "12:30"
          const match = res.data.answer.match(/(\d{1,2}):(\d{2})/);
          if (match) {
            refTime = parseInt(match[1]) * 60 + parseInt(match[2]);
          }

          setMessages(prev => [...prev, {
            sender: 'bot',
            avatarIcon: 'auto_awesome',
            text: res.data.answer,
            referenceTime: refTime
          }]);
        } else {
          setMessages(prev => [...prev, {
            sender: 'bot',
            avatarIcon: 'auto_awesome',
            text: 'Không nhận được câu trả lời từ hệ thống RAG.'
          }]);
        }
        setIsTyping(false);
      })
      .catch(err => {
        setMessages(prev => [...prev, {
          sender: 'bot',
          avatarIcon: 'auto_awesome',
          text: `Có lỗi xảy ra khi truy vấn ChromaDB: ${err.message || 'Lỗi kết nối'}`
        }]);
        setIsTyping(false);
      });
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
        text: 'Lịch sử trò chuyện đã được xóa. Tôi sẵn sàng trả lời các câu hỏi mới!'
      }
    ]);
  };

  return (
    <div className="flex flex-1 overflow-hidden h-[calc(100vh-64px)] bg-background text-on-surface">
      {/* Left Sidebar: Video & Key Segments */}
      <section className="w-full md:w-80 lg:w-96 bg-surface border-r border-outline-variant flex flex-col h-full shrink-0">
        {/* Video Player Container */}
        <div className="p-4 border-b border-outline-variant/30">
          <div className="relative aspect-video bg-video-background rounded-xl overflow-hidden border border-outline-variant shadow-sm">
            <video 
              ref={videoRef}
              src={videoData?.filePath ? (videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`) : ''} 
              poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
              controls
              className="w-full h-full object-cover"
            />
          </div>
          <div className="mt-4">
            <h2 className="font-headline-md text-base font-bold text-deep-navy">
              {videoData?.originalUrl ? 'YouTube Lecture' : (videoData?.filePath?.split('/').pop() || 'Lecture Materials')}
            </h2>
            <p className="font-body-sm text-xs text-secondary mt-1">
              Thời lượng: {videoData?.duration ? formatTimeText(videoData.duration) : '45:10'} • AI RAG Engine v2.4.0
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
            {/* Segment Card 1 */}
            <div 
              onClick={() => seekMiniPlayer(15)}
              className="p-3 rounded-lg border border-outline-variant hover:border-vibrant-cyan transition-colors cursor-pointer group"
            >
              <div className="flex justify-between items-start mb-2">
                <span className="bg-surface-container-highest px-2 py-0.5 rounded text-[10px] font-mono-data text-deep-navy">00:15 - 02:45</span>
                <span className="material-symbols-outlined text-vibrant-cyan text-sm opacity-0 group-hover:opacity-100">shortcut</span>
              </div>
              <h4 className="font-label-md text-xs font-bold text-deep-navy mb-1">Giới thiệu về Cross-modal Attention</h4>
              <p className="text-[11px] text-secondary line-clamp-2">Định nghĩa các điểm nghẽn kiến trúc trong việc xử lý đồng bộ video và audio.</p>
            </div>

            {/* Segment Card 2 */}
            <div 
              onClick={() => seekMiniPlayer(165)}
              className="p-3 rounded-lg border border-outline-variant hover:border-vibrant-cyan transition-colors cursor-pointer group"
            >
              <div className="flex justify-between items-start mb-2">
                <span className="bg-surface-container-highest px-2 py-0.5 rounded text-[10px] font-mono-data text-deep-navy">02:45 - 05:12</span>
                <span className="material-symbols-outlined text-vibrant-cyan text-sm opacity-0 group-hover:opacity-100">shortcut</span>
              </div>
              <h4 className="font-label-md text-xs font-bold text-deep-navy mb-1">Cơ chế Temporal Consistency</h4>
              <p className="text-[11px] text-secondary line-clamp-2">Đảm bảo tính nhất quán không gian giữa các timeframe sử dụng kỹ thuật latent interpolation.</p>
            </div>

            {/* Segment Card 3 */}
            <div 
              onClick={() => seekMiniPlayer(312)}
              className="p-3 rounded-lg border border-outline-variant hover:border-vibrant-cyan transition-colors cursor-pointer group"
            >
              <div className="flex justify-between items-start mb-2">
                <span className="bg-surface-container-highest px-2 py-0.5 rounded text-[10px] font-mono-data text-deep-navy">05:12 - 08:30</span>
                <span className="material-symbols-outlined text-vibrant-cyan text-sm opacity-0 group-hover:opacity-100">shortcut</span>
              </div>
              <h4 className="font-label-md text-xs font-bold text-deep-navy mb-1">Empirical Results &amp; Benchmarks</h4>
              <p className="text-[11px] text-secondary line-clamp-2">So sánh hiệu năng của tầng fusion với các mô hình ResNet-50 và ViT.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Chat Interface */}
      <section className="flex-1 flex flex-col relative h-full">
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
                  {msg.sender === 'user' ? 'Bạn' : 'Trợ lý AI'}
                </span>
                {msg.sender === 'bot' && idx > 0 && (
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
            </div>
          ))}

          {isTyping && (
            <div className="flex flex-col items-start">
              <div className="flex items-center gap-2 mb-1 px-2">
                <span className="font-label-sm text-xs text-deep-navy font-bold">Trợ lý AI</span>
              </div>
              <div className="glass-chat-ai p-4 rounded-2xl rounded-tl-none flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-vibrant-cyan rounded-full animate-bounce"></div>
                <div className="w-1.5 h-1.5 bg-vibrant-cyan rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-1.5 h-1.5 bg-vibrant-cyan rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <span className="text-xs text-secondary italic ml-2">Đang suy nghĩ...</span>
              </div>
            </div>
          )}
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
