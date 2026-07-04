import React, { useState, useRef, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { CONFIG } from '../config';
import './QaPage.css';

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
      avatarIcon: 'fa-solid fa-brain',
      text: 'Chào bạn! Hệ thống RAG đã được nạp toàn bộ Transcript và thông tin Keyframes (BLIP-2) của video. Bạn có muốn hỏi gì về bài giảng này không?'
    },
    {
      sender: 'user',
      avatarIcon: 'fa-solid fa-user',
      text: 'Giảng viên định nghĩa thế nào về kiến trúc Transformer? Tại sao nó tốt hơn RNN?'
    },
    {
      sender: 'bot',
      avatarIcon: 'fa-solid fa-brain',
      text: 'Theo bài giảng, kiến trúc Transformer là một mô hình học sâu chủ yếu dựa trên cơ chế <strong>Self-Attention</strong>. Giảng viên giải thích rằng khác với RNN (Recurrent Neural Networks) phải xử lý dữ liệu tuần tự từng từ một, Transformer có khả năng xem xét toàn bộ câu cùng một lúc.<br/><br/>Điều này mang lại hai lợi ích chính:<ul style="margin-top: 10px; padding-left: 20px;"><li>Khắc phục triệt để vấn đề mất mát thông tin đối với chuỗi văn bản dài.</li><li>Cho phép tính toán song song trên GPU, giúp tốc độ huấn luyện nhanh hơn rất nhiều.</li></ul>',
      referenceTime: 920 // 15:20 in seconds
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

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const queryText = inputText;
    const newMsg: Message = {
      sender: 'user',
      avatarIcon: 'fa-solid fa-user',
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
            avatarIcon: 'fa-solid fa-brain',
            text: res.data.answer,
            referenceTime: refTime
          }]);
        } else {
          setMessages(prev => [...prev, {
            sender: 'bot',
            avatarIcon: 'fa-solid fa-brain',
            text: 'Không nhận được câu trả lời từ hệ thống RAG.'
          }]);
        }
        setIsTyping(false);
      })
      .catch(err => {
        setMessages(prev => [...prev, {
          sender: 'bot',
          avatarIcon: 'fa-solid fa-brain',
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
        avatarIcon: 'fa-solid fa-brain',
        text: 'Lịch sử trò chuyện đã được xóa. Tôi sẵn sàng trả lời các câu hỏi mới!'
      }
    ]);
  };

  return (
    <div className="qa-page">
      <div className="sidebar">
        <div className="sidebar-header">
          <Link to={`/results?videoId=${videoId}`} className="back-link"><i className="fa-solid fa-arrow-left"></i></Link>
          <h2>Ngữ Cảnh Video (RAG)</h2>
        </div>
        <div className="sidebar-content">
          <div className="video-mini-container">
            <video 
              ref={videoRef}
              src={videoData?.filePath ? (videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`) : ''} 
              poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
              controls
              className="video-mini"
            />
          </div>
          <div className="info-card">
            <p>Tên file: <span>{videoData?.filePath ? videoData.filePath.split('/').pop() : videoData?.originalUrl ? 'YouTube Source' : 'MIT_DeepLearning_Lec1.mp4'}</span></p>
            <p>Thời lượng: <span>{videoData?.duration ? formatTimeText(videoData.duration) : '45:10'}</span></p>
            <p>Vector Store: <span>ChromaDB</span></p>
            <p>LLM Backend: <span>Groq Cloud</span></p>
          </div>
          
          <div className="history-list">
            <h3>Câu hỏi gợi ý</h3>
            <div className="history-item" onClick={() => setInputText('Giải thích Backpropagation')}>
              <i className="fa-regular fa-message" style={{ marginRight: '10px' }}></i> Giải thích Backpropagation
            </div>
            <div className="history-item" onClick={() => setInputText('Hạn chế của mạng RNN là gì?')}>
              <i className="fa-regular fa-message" style={{ marginRight: '10px' }}></i> Hạn chế của mạng RNN
            </div>
            <div className="history-item" onClick={() => setInputText('Cơ chế của Transformer?')}>
              <i className="fa-regular fa-message" style={{ marginRight: '10px' }}></i> Cơ chế của Transformer
            </div>
          </div>
        </div>
      </div>
      
      <div className="chat-area">
        <div className="chat-header">
          <h3 style={{ fontWeight: 500, fontSize: '1.2rem', margin: 0, color: 'var(--text-main)' }}>
            <i className="fa-solid fa-robot" style={{ color: 'var(--primary)', marginRight: '10px' }}></i> Trợ lý Video AI
          </h3>
          <button className="btn text-btn" onClick={clearHistory}>
            <i className="fa-solid fa-trash-can"></i> Xóa lịch sử
          </button>
        </div>
        
        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`msg ${msg.sender === 'user' ? 'user' : 'bot'}`}>
              <div className="avatar">
                <i className={msg.avatarIcon}></i>
              </div>
              <div className="bubble-wrapper">
                <div 
                  className="bubble"
                  dangerouslySetInnerHTML={{ __html: msg.text }}
                />
                {msg.referenceTime !== undefined && (
                  <button 
                    className="ref-badge"
                    onClick={() => seekMiniPlayer(msg.referenceTime!)}
                  >
                    <i className="fa-solid fa-play"></i> Xem đoạn: {formatTimeText(msg.referenceTime)}
                  </button>
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="msg bot">
              <div className="avatar"><i className="fa-solid fa-brain"></i></div>
              <div className="bubble loading">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
        </div>
        
        <div className="input-area">
          <div className="input-box">
            <input 
              type="text" 
              placeholder="Nhập câu hỏi của bạn để tìm kiếm trong video (Enter để gửi)..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button className="send-btn" onClick={handleSendMessage}>
              <i className="fa-solid fa-paper-plane"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
