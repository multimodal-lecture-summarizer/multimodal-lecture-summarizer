import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import './QaPage.css';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp?: string;
  avatarIcon: string;
  referenceTime?: number; // Time in seconds to seek to
}

export const QaPage: React.FC = () => {
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

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const newMsg: Message = {
      sender: 'user',
      avatarIcon: 'fa-solid fa-user',
      text: inputText
    };

    setMessages(prev => [...prev, newMsg]);
    setInputText('');
    setIsTyping(true);

    // Mock bot response after 1.5 seconds
    setTimeout(() => {
      let botResponse = 'Tôi đã tìm kiếm câu hỏi này trong video. ';
      let refTime = 0;
      
      const q = inputText.toLowerCase();
      if (q.includes('backpropagation') || q.includes('lan truyền ngược') || q.includes('truyền ngược')) {
        botResponse = 'Theo video giảng dạy, <strong>Backpropagation</strong> là xương sống của việc tối ưu hóa mạng nơ-ron. Thuật toán hoạt động bằng cách tính toán đạo hàm (gradient) của hàm mất mát cho tất cả các trọng số trong mạng nơ-ron theo chu kỳ ngược từ layer đầu ra về layer đầu vào. Sau đó, các trọng số được cập nhật thông qua Gradient Descent để tìm điểm cực tiểu.';
        refTime = 312; // 05:12
      } else if (q.includes('rnn') || q.includes('hạn chế')) {
        botResponse = 'Giảng viên đề cập rằng mạng RNN truyền thống gặp khó khăn rất lớn khi huấn luyện trên các chuỗi văn bản dài. Vấn đề cốt lõi là hiện tượng <strong>tiêu biến gradient (vanishing gradient)</strong>, khiến các layer thời điểm đầu không học được thông tin từ các layer cuối. Do đó, RNN không giữ được ngữ cảnh dài.';
        refTime = 750; // 12:30
      } else if (q.includes('giới thiệu') || q.includes('bài giảng')) {
        botResponse = 'Ở đầu bài giảng, giảng viên chào mừng các học sinh và giới thiệu tổng quan về cấu trúc khóa học Trí tuệ Nhân tạo và vị trí của Học sâu (Deep Learning) trong dòng chảy phát triển của khoa học máy tính.';
        refTime = 0; // 00:00
      } else {
        botResponse = `Cơ sở dữ liệu Vector (ChromaDB) đã tìm thấy thông tin trùng khớp với truy vấn của bạn. Đoạn hội thoại này xuất hiện ở khoảng mốc thời gian ${formatTimeText(312)}. Giảng viên đang giải thích khái niệm chính của bài học. Bạn có thể nhấn badge bên dưới để theo dõi trực tiếp.`;
        refTime = 312;
      }

      setMessages(prev => [...prev, {
        sender: 'bot',
        avatarIcon: 'fa-solid fa-brain',
        text: botResponse,
        referenceTime: refTime
      }]);
      setIsTyping(false);
    }, 1500);
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
          <Link to="/results" className="back-link"><i className="fa-solid fa-arrow-left"></i></Link>
          <h2>Ngữ Cảnh Video (RAG)</h2>
        </div>
        <div className="sidebar-content">
          <div className="video-mini-container">
            <video 
              ref={videoRef}
              src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4" 
              poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
              controls
              className="video-mini"
            />
          </div>
          <div className="info-card">
            <p>Tên file: <span>MIT_DeepLearning_Lec1.mp4</span></p>
            <p>Thời lượng: <span>45:10</span></p>
            <p>Vector Store: <span>ChromaDB</span></p>
            <p>LLM Backend: <span>Qwen2.5-7B (Local)</span></p>
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
