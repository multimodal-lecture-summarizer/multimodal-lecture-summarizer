import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';
import './AdminPage.css';

export const AdminPage: React.FC = () => {
  const barChartRef = useRef<HTMLCanvasElement>(null);
  const doughnutChartRef = useRef<HTMLCanvasElement>(null);
  const barInstance = useRef<Chart | null>(null);
  const doughnutInstance = useRef<Chart | null>(null);

  useEffect(() => {
    // Destroy existing instances if any (essential for React HMR)
    if (barInstance.current) barInstance.current.destroy();
    if (doughnutInstance.current) doughnutInstance.current.destroy();

    // Render Bar Chart
    if (barChartRef.current) {
      barInstance.current = new Chart(barChartRef.current, {
        type: 'bar',
        data: {
          labels: ['Tuần 1', 'Tuần 2', 'Tuần 3', 'Tuần 4', 'Tuần 5', 'Tuần 6'],
          datasets: [{
            label: 'Số Video Xử Lý',
            data: [320, 450, 680, 890, 1200, 1540],
            backgroundColor: '#4f46e5',
            borderRadius: 6,
            barThickness: 20
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              grid: { color: '#e2e8f0' },
              beginAtZero: true
            },
            x: {
              grid: { display: false }
            }
          }
        }
      });
    }

    // Render Doughnut Chart
    if (doughnutChartRef.current) {
      doughnutInstance.current = new Chart(doughnutChartRef.current, {
        type: 'doughnut',
        data: {
          labels: ['GPT-4o API', 'Gemini 1.5 API', 'Qwen2.5 Local'],
          datasets: [{
            data: [30, 25, 45],
            backgroundColor: ['#3b82f6', '#ec4899', '#10b981'],
            borderWidth: 0,
            hoverOffset: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                padding: 15,
                usePointStyle: true,
                font: { size: 11 }
              }
            }
          },
          cutout: '75%'
        }
      });
    }

    return () => {
      if (barInstance.current) barInstance.current.destroy();
      if (doughnutInstance.current) doughnutInstance.current.destroy();
    };
  }, []);

  return (
    <div className="admin-page animate-fade-in">
      <div className="admin-sidebar">
        <div className="admin-logo">
          <i className="fa-solid fa-shield-halved"></i> Dashboard
        </div>
        
        <div className="admin-nav-group">
          <div className="admin-nav-label">Analytics</div>
          <a href="#stats" className="admin-nav-item active">
            <i className="fa-solid fa-chart-pie"></i> Báo cáo hệ thống
          </a>
          <a href="#metrics" className="admin-nav-item">
            <i className="fa-solid fa-chart-line"></i> Hiệu suất AI
          </a>
        </div>
        
        <div className="admin-nav-group">
          <div className="admin-nav-label">Quản lý</div>
          <a href="#users" className="admin-nav-item">
            <i className="fa-solid fa-users"></i> Người dùng
          </a>
          <a href="#videos" className="admin-nav-item">
            <i className="fa-solid fa-video"></i> Tiêu chuẩn Video
          </a>
          <a href="#celery" className="admin-nav-item">
            <i className="fa-solid fa-server"></i> Celery Job Queue
          </a>
        </div>
      </div>
      
      <div className="admin-main-content">
        <div className="admin-header">
          <h1>Báo Cáo Thống Kê Tổng Hợp</h1>
          <div className="calendar-selector">
            <i className="fa-regular fa-calendar"></i> Tháng 6, 2026 <i className="fa-solid fa-chevron-down" style={{ marginLeft: '10px', fontSize: '0.8rem' }}></i>
          </div>
        </div>
        
        <div className="admin-stats-grid">
          <div className="admin-stat-card">
            <h3>Người dùng <i className="fa-solid fa-users" style={{ color: 'var(--primary)' }}></i></h3>
            <div className="value">2,540</div>
            <div className="trend up"><i className="fa-solid fa-arrow-trend-up"></i> +12.5%</div>
          </div>
          <div className="admin-stat-card">
            <h3>Video Xử Lý <i className="fa-solid fa-film" style={{ color: 'var(--primary)' }}></i></h3>
            <div className="value">8,932</div>
            <div className="trend up"><i className="fa-solid fa-arrow-trend-up"></i> +8.2%</div>
          </div>
          <div className="admin-stat-card">
            <h3>Audio WER <i className="fa-solid fa-microphone-lines" style={{ color: '#ec4899' }}></i></h3>
            <div className="value">7.8%</div>
            <div className="trend label-ok"><i className="fa-solid fa-check"></i> Đạt chuẩn &lt; 10%</div>
          </div>
          <div className="admin-stat-card">
            <h3>Keyframe F-score <i className="fa-solid fa-image" style={{ color: '#3b82f6' }}></i></h3>
            <div className="value">0.52</div>
            <div className="trend label-ok"><i className="fa-solid fa-check"></i> Vượt mục tiêu (0.45)</div>
          </div>
        </div>
        
        <div className="admin-charts-row">
          <div className="admin-chart-card">
            <h3>Lưu lượng Video (Job Queue)</h3>
            <div className="canvas-wrapper">
              <canvas ref={barChartRef}></canvas>
            </div>
          </div>
          <div className="admin-chart-card">
            <h3>Sử dụng Mô hình LLM</h3>
            <div className="canvas-wrapper doughnut-wrapper">
              <canvas ref={doughnutChartRef}></canvas>
            </div>
          </div>
        </div>
        
        <div className="admin-table-card">
          <h3 className="admin-table-title">Tác vụ Xử lý Gần Đây (PostgreSQL `jobs` table)</h3>
          <div className="admin-table-container">
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Email User</th>
                  <th>Tên File / Nguồn</th>
                  <th>Độ dài</th>
                  <th>Trạng thái</th>
                  <th>AI Pipeline</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="uuid">#UUID-A8F9</td>
                  <td>nguyen.a@gmail.com</td>
                  <td>MIT 6.S191 - Deep Learning</td>
                  <td>45:10</td>
                  <td><span className="status done">Hoàn tất (2m 15s)</span></td>
                  <td>WhisperX + GPT-4o</td>
                </tr>
                <tr>
                  <td className="uuid">#UUID-B2E1</td>
                  <td>tran.b@student.edu.vn</td>
                  <td>TED Talk - Future of AI</td>
                  <td>12:45</td>
                  <td><span className="status processing">Visual Module (CLIP)</span></td>
                  <td>WhisperX + Gemini</td>
                </tr>
                <tr>
                  <td className="uuid">#UUID-C4D5</td>
                  <td>le.c@company.com</td>
                  <td>Internal Training 2026.mp4</td>
                  <td>1:20:00</td>
                  <td><span className="status failed">Lỗi: Vượt quá 60 phút</span></td>
                  <td>-</td>
                </tr>
                <tr>
                  <td className="uuid">#UUID-D7F2</td>
                  <td>admin@system.local</td>
                  <td>Stanford CS224N - NLP</td>
                  <td>55:30</td>
                  <td><span className="status done">Hoàn tất (3m 05s)</span></td>
                  <td>WhisperX + Qwen2.5 Local</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
