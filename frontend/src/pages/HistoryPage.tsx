import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { HistoryItem } from '../types';
import { VideoStatus } from '../types';
import { api } from '../services/api';

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();

  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([
    {
      id: 'mit-quantum-04',
      title: 'Quantum Mechanics Lecture 04: Wave-Particle Duality',
      duration: '42:15',
      date: '24/10/2024, 09:30',
      status: VideoStatus.DONE
    },
    {
      id: 'neuroscience-learning',
      title: 'Neural Plasticity & Cognitive Load Theory',
      duration: '58:40',
      date: '22/10/2024, 14:15',
      status: VideoStatus.DONE
    },
    {
      id: 'renewable-energy',
      title: 'Renewable Energy Systems & Global Economics',
      duration: '1:12:00',
      date: '19/10/2024, 11:20',
      status: VideoStatus.DONE
    },
    {
      id: 'biochem-crispr',
      title: 'CRISPR-Cas9: Ethics & Application',
      duration: '35:20',
      date: '15/10/2024, 08:45',
      status: VideoStatus.DONE
    },
    {
      id: 'urban-mobility',
      title: 'Urban Mobility: 15-Minute City Simulation',
      duration: '1:45:10',
      date: '10/10/2024, 16:30',
      status: VideoStatus.DONE
    }
  ]);

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSubject, setSelectedSubject] = useState('All');
  const [sortBy, setSortBy] = useState('newest');

  useEffect(() => {
    api.getVideos()
      .then(res => {
        if (res.success && res.data && res.data.length > 0) {
          const items = res.data.map((video: any) => {
            const durationSec = video.duration || 0;
            const hours = Math.floor(durationSec / 3600);
            const minutes = Math.floor((durationSec % 3600) / 60);
            const seconds = Math.floor(durationSec % 60);
            
            const durationStr = hours > 0 
              ? `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
              : `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            const uploadDate = new Date(video.uploadedAt);
            const dateStr = uploadDate.toLocaleDateString('vi-VN') + ', ' + uploadDate.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });

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

  // Helper to assign mock subjects, images, and accuracy scores
  const getCardDetails = (title: string, index: number) => {
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes('quantum') || lowerTitle.includes('duality')) {
      return {
        subject: 'Quantum Physics',
        bgImage: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBOgev44yFeRiOsw4c7SmUkCik71FMKLVFPmKyFMlIecVA_MX6oicypP5d-qtIoqq_HTw_xZVxaq6cDKqHxj3RSGcRFC-V7lGfYeVRyh1nw-wBLYSptMS5cR-GrgOCx0gOHp2RnYcE6P8e3wGuoDsU_-lNYcoQTPLeJDh8wrHedAVF4rE4aAAFepFhv0ZHUwJBpxR0UuBDWEb97soNoOunFeAxE6HC3p4ZEg0Rs9jU3JMiVQsenpRfl',
        accuracy: 98
      };
    } else if (lowerTitle.includes('neuro') || lowerTitle.includes('plasticity') || lowerTitle.includes('cognitive') || lowerTitle.includes('learning')) {
      return {
        subject: 'Neuroscience',
        bgImage: 'https://lh3.googleusercontent.com/aida-public/AB6AXuC6UZ4SQLSuBfiHk0wnEjZh1yQTQayOgdlHWV2kwLhVruVaDj3D6SAbHtbFtnYo2Ts1wXEHMmExJuyFpGaEHAPkRaEU6LdArBiZ__zZdOMQaNy2WDdPKpWDocdJ6Cl80Wz_yMqPwbHL4KNPNIEqQG3-V4N1ZPviDvoIO2VsGweI8MDP4jmQ_6aDTuQsA2gMfIswi_duiMOaThlaoW6jfMKfNwhvd4WPijkOQTqvIYUQMI1QzJ1dJ7Nu',
        accuracy: 94
      };
    } else if (lowerTitle.includes('renewable') || lowerTitle.includes('energy') || lowerTitle.includes('economics') || lowerTitle.includes('policy')) {
      return {
        subject: 'Environmental Policy',
        bgImage: 'https://lh3.googleusercontent.com/aida-public/AB6AXuB5WjqdYmKjA32gYcPClvTv8qoYHGg9QMTLM6BYAQyUqtXHz5OyPjQoQNbNVUQ2jHmhfvgVxF4SVNVXB81S3waTC9r3TJQt__OSvtzWEPVxB3QK4strC1cAJ424Ubp33OacveKVxOaPUiikN7ATWwX2Q-zSiw8YuhHXWZqDHkpKgWbDN2JLAfWsgLFyu0Cg24uIng0ebWhbKfRz1UHyeJczr88PlhEnN75IMdQra69rUyueTZcRBhmj',
        accuracy: 82
      };
    } else if (lowerTitle.includes('crispr') || lowerTitle.includes('cas9') || lowerTitle.includes('ethics') || lowerTitle.includes('biochemistry')) {
      return {
        subject: 'Biochemistry',
        bgImage: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDDrCT8ujApHRuFt3J3XYRbiY4KfDv0QbXjY1bmsUbhXdfz7yKPweu7MVzv95nZO8ZQqlzmKjgt228emNbvLu0tBu85lSN_lDT-A0dIHqDdpV0uaZFTE-u6Z9noGhg8EkVFcMgtxNADDuquWOBui-VojlWUCVElWEYwZ3QRa6M9iDroBzDHkVXUh7Hl5Kxis8rIVlMJAi_4gK2XwFeEbdikhXjPJPCjsAIIyMGdvX9Xr5MyTmbNI_6F',
        accuracy: 99
      };
    } else if (lowerTitle.includes('urban') || lowerTitle.includes('mobility') || lowerTitle.includes('city') || lowerTitle.includes('design')) {
      return {
        subject: 'Urban Design',
        bgImage: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDLlUKF6B4yFHqlfO3vUAdNgkfpQBxAYwiYU-JVnklFgXneLFDpdaz5mXxbC4meRPeAPPViBFvVkW7KCnbX3K-svYjHSPO9pfNLfbZ_eO0HJHY99-GGbkkYdoijSME5CGNM032a6STRwNRB9iQa8kcDRtg6bblN2kXxSH4fvVsWG76FvOCRLi4jVzmwL0CMkFYKaSe0UuwNJssVTzSs7WhohoU6AW3ovjkTHy3-SeNxEUYsko9nWS1e',
        accuracy: 96
      };
    } else {
      const fallbacks = [
        {
          subject: 'General Science',
          bgImage: 'https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60',
          accuracy: 90
        },
        {
          subject: 'Artificial Intelligence',
          bgImage: 'https://images.unsplash.com/photo-1677442136019-21780efad99a?w=800&auto=format&fit=crop&q=60',
          accuracy: 95
        }
      ];
      return fallbacks[index % fallbacks.length];
    }
  };

  // Filter & Sort
  const filteredItems = historyItems
    .filter(item => {
      const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase());
      const details = getCardDetails(item.title, 0);
      const matchesSubject = selectedSubject === 'All' || details.subject === selectedSubject;
      return matchesSearch && matchesSubject;
    })
    .sort((_a, _b) => {
      if (sortBy === 'newest') return 1;
      if (sortBy === 'oldest') return -1;
      return 0;
    });

  const queuedCount = historyItems.filter(item => item.status === VideoStatus.PROCESSING).length;
  const processedCount = historyItems.filter(item => item.status === VideoStatus.DONE).length;

  return (
    <div className="bg-surface text-on-surface font-body-md text-body-md antialiased min-h-screen">
      {/* Main Content Canvas */}
      <main className="flex-1 px-margin-mobile md:px-margin-desktop py-8 max-w-container-max mx-auto">
        {/* Page Header & Filtering Section */}
        <header className="mb-10">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <div>
              <h1 className="font-headline-xl text-3xl md:text-headline-xl text-deep-navy font-bold mb-2">Lịch Sử Phân Tích</h1>
              <p className="text-secondary max-w-2xl font-body-md text-sm md:text-body-md">
                Xem lại, quản lý và tra cứu các bản tóm tắt học thuật từ kho lưu trữ dữ liệu nghiên cứu đa phương thức của bạn.
              </p>
            </div>
            <div className="flex items-center gap-4 text-mono-data font-mono-data text-xs bg-surface-container-high px-4 py-2 rounded-full border border-outline-variant shrink-0">
              <span className="flex items-center gap-1 font-bold">
                <span className="w-2 h-2 rounded-full bg-status-success"></span> {processedCount} Hoàn tất
              </span>
              <span className="text-outline-variant">|</span>
              <span className="flex items-center gap-1 font-bold">
                <span className="w-2 h-2 rounded-full bg-status-warning animate-pulse"></span> {queuedCount} Đang xử lý
              </span>
            </div>
          </div>

          {/* Bento-style Filter Bar */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-4 lg:grid-cols-5 gap-4">
            <div className="md:col-span-2 relative">
              <input 
                className="w-full pl-10 pr-4 py-3 bg-white border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none transition-all font-body-sm text-sm shadow-sm" 
                placeholder="Tìm kiếm tiêu đề hoặc từ khóa bài giảng..." 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <span className="absolute left-3 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline text-xl">manage_search</span>
            </div>
            
            <select 
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-white border border-outline-variant rounded-xl px-4 py-3 text-sm focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none cursor-pointer"
            >
              <option value="newest">Sắp xếp: Mới nhất</option>
              <option value="oldest">Sắp xếp: Cũ nhất</option>
            </select>

            <select 
              value={selectedSubject}
              onChange={(e) => setSelectedSubject(e.target.value)}
              className="bg-white border border-outline-variant rounded-xl px-4 py-3 text-sm focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none cursor-pointer"
            >
              <option value="All">Tất cả chủ đề</option>
              <option value="Quantum Physics">Quantum Physics</option>
              <option value="Neuroscience">Neuroscience</option>
              <option value="Environmental Policy">Environmental Policy</option>
              <option value="Biochemistry">Biochemistry</option>
              <option value="Urban Design">Urban Design</option>
            </select>

            <button className="bg-white border border-outline-variant rounded-xl px-4 py-3 text-sm flex items-center justify-center gap-2 hover:bg-surface-container-low transition-colors font-medium">
              <span className="material-symbols-outlined text-secondary text-lg">tune</span>
              Bộ lọc nâng cao
            </button>
          </div>
        </header>

        {/* Analysis Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-gutter-desktop">
          {filteredItems.map((item, index) => {
            const details = getCardDetails(item.title, index);
            return (
              <div key={item.id} className="group bg-white border border-outline-variant rounded-xl overflow-hidden hover:border-vibrant-cyan transition-all flex flex-col shadow-sm">
                <div className="relative h-48 w-full overflow-hidden bg-video-background shrink-0">
                  <div className="absolute inset-0 bg-black/40 group-hover:bg-black/20 transition-colors z-10"></div>
                  <img 
                    className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500 opacity-80" 
                    alt={item.title}
                    src={details.bgImage}
                  />
                  <div className="absolute top-3 left-3 z-20 px-2 py-1 bg-deep-navy/80 text-white rounded font-mono-data text-[10px] uppercase tracking-wider backdrop-blur-md">
                    {details.subject}
                  </div>
                  <div className="absolute bottom-3 right-3 z-20 px-2 py-1 bg-status-success text-white rounded-full flex items-center gap-1 font-label-sm text-[11px] font-bold shadow-md">
                    <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span> 
                    {details.accuracy}% Accuracy
                  </div>
                </div>

                <div className="p-5 flex-grow flex flex-col justify-between">
                  <div className="mb-4 space-y-2">
                    <h3 className="font-headline-md text-base font-bold text-deep-navy leading-tight group-hover:text-vibrant-cyan transition-colors line-clamp-2" title={item.title}>
                      {item.title}
                    </h3>
                    <div className="text-secondary font-body-sm text-xs flex items-center gap-3">
                      <span className="flex items-center gap-1">
                        <span className="material-symbols-outlined text-sm">calendar_today</span> 
                        {item.date.split(',')[0]}
                      </span>
                      <span className="text-outline-variant">•</span>
                      <span className="flex items-center gap-1">
                        <span className="material-symbols-outlined text-sm">schedule</span> 
                        {item.duration}
                      </span>
                    </div>
                  </div>

                  <div className="mt-auto flex items-center justify-between pt-4 border-t border-outline-variant/30 shrink-0">
                    <div className="flex gap-2">
                      {item.status === VideoStatus.DONE ? (
                        <Link 
                          to={`/results?videoId=${item.id}`}
                          className="w-10 h-10 flex items-center justify-center text-secondary hover:text-vibrant-cyan hover:bg-surface-container-low rounded-lg border border-outline-variant/50 transition-all active:scale-95" 
                          title="Xem kết quả"
                        >
                          <span className="material-symbols-outlined text-xl">visibility</span>
                        </Link>
                      ) : (
                        <button 
                          disabled
                          className="w-10 h-10 flex items-center justify-center text-slate-400 bg-slate-100 rounded-lg cursor-not-allowed border border-slate-200" 
                          title="Đang phân tích..."
                        >
                          <span className="material-symbols-outlined text-xl animate-spin">autorenew</span>
                        </button>
                      )}
                      
                      <button 
                        onClick={() => {
                          if (item.status === VideoStatus.DONE) {
                            api.exportSummary(item.id, 'pdf')
                              .then(blob => {
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `summary_${item.id}.pdf`;
                                document.body.appendChild(a);
                                a.click();
                                a.remove();
                              })
                              .catch(err => alert(`Lỗi tải PDF: ${err.message}`));
                          }
                        }}
                        className="w-10 h-10 flex items-center justify-center text-secondary hover:text-vibrant-cyan hover:bg-surface-container-low rounded-lg border border-outline-variant/50 transition-all active:scale-95" 
                        title="Tải PDF"
                        disabled={item.status !== VideoStatus.DONE}
                      >
                        <span className="material-symbols-outlined text-xl">picture_as_pdf</span>
                      </button>
                    </div>
                    
                    <button 
                      onClick={() => handleDelete(item.id)}
                      className="w-10 h-10 flex items-center justify-center text-secondary hover:text-error hover:bg-error-container/30 rounded-lg border border-outline-variant/50 transition-all active:scale-95" 
                      title="Xóa"
                    >
                      <span className="material-symbols-outlined text-xl">delete_outline</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Empty State Suggestion Card (Upload card) */}
          <div 
            onClick={() => navigate('/upload')}
            className="border-2 border-dashed border-outline-variant rounded-xl flex flex-col items-center justify-center p-8 text-center bg-white hover:bg-surface-container-low transition-colors cursor-pointer group shadow-sm min-h-[300px]"
          >
            <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <span className="material-symbols-outlined text-outline group-hover:text-vibrant-cyan text-3xl">cloud_upload</span>
            </div>
            <h4 className="font-headline-md text-sm font-bold text-deep-navy mb-2">Phân Tích Thêm Video</h4>
            <p className="text-secondary font-body-sm text-xs mb-6 max-w-[200px] leading-relaxed">Tải lên các bản ghi bài giảng đa phương thức mới để mở rộng kho kiến thức của bạn.</p>
            <button className="px-6 py-2.5 bg-deep-navy text-white hover:bg-slate-800 rounded-xl font-bold text-xs uppercase tracking-wider transition-all active:scale-95 shadow-sm">
              Upload Video
            </button>
          </div>
        </div>

        {/* Pagination */}
        <div className="mt-12 flex items-center justify-center gap-2">
          <button className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all disabled:opacity-30 active:scale-95" disabled>
            <span className="material-symbols-outlined text-xl">chevron_left</span>
          </button>
          <button className="w-10 h-10 flex items-center justify-center bg-deep-navy text-white rounded-lg font-bold text-xs active:scale-95 shadow-sm">1</button>
          <button className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all font-bold text-xs active:scale-95">2</button>
          <button className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all font-bold text-xs active:scale-95">3</button>
          <span className="px-2 text-outline-variant">...</span>
          <button className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all font-bold text-xs active:scale-95">12</button>
          <button className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all active:scale-95">
            <span className="material-symbols-outlined text-xl">chevron_right</span>
          </button>
        </div>
      </main>
      
      {/* Footer Shell */}
      <footer className="bg-surface-container-lowest flex flex-col md:flex-row justify-between items-center px-margin-desktop py-8 w-full border-t border-outline-variant mt-12 text-xs">
        <div className="mb-4 md:mb-0 text-center md:text-left space-y-1">
          <span className="font-label-md font-bold text-deep-navy">Multimodal Lecture Summarizer</span>
          <p className="text-secondary font-body-sm">© 2026 Institute for Multimodal AI Research. All rights reserved.</p>
        </div>
        <div className="flex flex-wrap justify-center gap-6 font-semibold">
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#credits">Academic Credits</a>
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#privacy">Privacy Policy</a>
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#terms">Terms of Service</a>
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#docs">Research Documentation</a>
        </div>
      </footer>
    </div>
  );
};
