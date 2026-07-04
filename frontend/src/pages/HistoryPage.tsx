import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import type { HistoryItem } from '../types';
import { VideoStatus } from '../types';
import { api } from '../services/api';
import './HistoryPage.css';

export const HistoryPage: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([
    {
      id: 'mit-dl-1',
      title: 'Bài giảng MIT 6.S191 - Introduction to Deep Learning',
      duration: '45:10',
      date: 'Hôm nay, 10:30',
      status: VideoStatus.DONE
    },
    {
      id: 'ted-ai',
      title: 'TED Talk - Tương lai của Trí tuệ Nhân tạo',
      duration: '12:45',
      date: 'Hôm qua, 15:20',
      status: VideoStatus.PROCESSING
    },
    {
      id: 'cs224n-nlp',
      title: 'CS224N - NLP with Deep Learning',
      duration: '55:30',
      date: '25/06/2026',
      status: VideoStatus.DONE
    }
  ]);

  useEffect(() => {
    api.getVideos()
      .then(res => {
        if (res.success && res.data && res.data.length > 0) {
          const items = res.data.map((video: any) => {
            const durationSec = video.duration || 0;
            const minutes = Math.floor(durationSec / 60);
            const seconds = Math.floor(durationSec % 60);
            const durationStr = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
            
            const uploadDate = new Date(video.uploadedAt);
            const dateStr = uploadDate.toLocaleDateString('vi-VN') + ', ' + uploadDate.toLocaleTimeString('vi-VN', {hour: '2-digit', minute:'2-digit'});

            return {
              id: video.videoId,
              title: video.originalUrl ? `YouTube: ${video.originalUrl}` : `Local Video: ${video.filePath?.split('/').pop() || 'Video#' + video.videoId.substring(0,6)}`,
              duration: durationStr,
              date: dateStr,
              status: video.status as VideoStatus,
            };
          });
          setHistoryItems(items);
        }
      })
      .catch(err => {
        console.warn("Failed to fetch real history from backend, showing mock data.", err);
      });
  }, []);

  const handleDelete = (id: string) => {
    if (window.confirm("Bạn có chắc chắn muốn xóa video này khỏi lịch sử?")) {
      setHistoryItems(prev => prev.filter(item => item.id !== id));
    }
  };

  return (
    <div className="history-page animate-fade-in">
      <div className="container">
        <h1>Lịch Sử Video Của Bạn</h1>
        
        {historyItems.length === 0 ? (
          <div className="empty-state">
            <i className="fa-solid fa-folder-open"></i>
            <p>Lịch sử của bạn đang trống. Hãy bắt đầu phân tích video mới!</p>
            <Link to="/upload" className="btn primary" style={{ marginTop: '15px' }}>
              <i className="fa-solid fa-plus"></i> Upload Video Mới
            </Link>
          </div>
        ) : (
          historyItems.map((item) => (
            <div key={item.id} className="history-card">
              <div className="thumbnail">
                {item.id === 'ted-ai' ? (
                  <i className="fa-brands fa-youtube" style={{ fontSize: '2rem' }}></i>
                ) : item.id === 'cs224n-nlp' ? (
                  <i className="fa-solid fa-video" style={{ fontSize: '2rem' }}></i>
                ) : (
                  <i className="fa-solid fa-image" style={{ fontSize: '2rem' }}></i>
                )}
              </div>
              
              <div className="info">
                <h3>{item.title}</h3>
                <div className="meta">
                  <span><i className="fa-regular fa-clock"></i> {item.duration}</span>
                  <span><i className="fa-regular fa-calendar"></i> {item.date}</span>
                </div>
                {item.status === VideoStatus.DONE ? (
                  <span className="status done"><i className="fa-solid fa-check"></i> Hoàn tất</span>
                ) : (
                  <span className="status processing">
                    <i className="fa-solid fa-spinner fa-spin"></i> Đang trích xuất Audio...
                  </span>
                )}
              </div>
              
              <div className="actions">
                {item.status === VideoStatus.DONE ? (
                  <Link to={`/results?videoId=${item.id}`} className="btn primary">
                    <i className="fa-regular fa-eye"></i> Xem Kết Quả
                  </Link>
                ) : (
                  <button className="btn" disabled style={{ opacity: 0.5 }}>
                    <i className="fa-regular fa-eye"></i> Đang Xử Lý
                  </button>
                )}
                <button className="btn danger" onClick={() => handleDelete(item.id)}>
                  <i className="fa-regular fa-trash-can"></i>
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
