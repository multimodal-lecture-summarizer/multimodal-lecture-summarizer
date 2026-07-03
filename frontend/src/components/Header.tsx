import React from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import './Header.css';

interface HeaderProps {
  isLoggedIn?: boolean;
  onLogout?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isLoggedIn = true, onLogout }) => {
  const location = useLocation();
  const isAdminPage = location.pathname.startsWith('/admin');

  return (
    <header className="app-header">
      {isAdminPage ? (
        <Link to="/admin" className="logo">
          <i className="fa-solid fa-shield-halved"></i> Admin Panel
        </Link>
      ) : (
        <Link to="/" className="logo">
          <i className="fa-solid fa-brain"></i> AI.Summarizer
        </Link>
      )}

      {!isAdminPage && (
        <nav className="nav-links">
          <NavLink to="/" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Trang chủ
          </NavLink>
          <NavLink to="/history" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Lịch sử video
          </NavLink>
          <NavLink to="/admin" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            Quản trị
          </NavLink>
          <a href="#docs" className="nav-item">Tài liệu</a>
        </nav>
      )}

      {isAdminPage && (
        <nav className="nav-links">
          <NavLink to="/admin" end className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
            <i className="fa-solid fa-chart-pie"></i> Báo cáo
          </NavLink>
          <NavLink to="/admin/metrics" className="nav-item">
            <i className="fa-solid fa-chart-line"></i> Hiệu suất AI
          </NavLink>
          <NavLink to="/admin/queue" className="nav-item">
            <i className="fa-solid fa-server"></i> Celery Queue
          </NavLink>
        </nav>
      )}

      <div className="header-actions">
        {isLoggedIn ? (
          <>
            {isAdminPage ? (
              <Link to="/" className="btn">
                <i className="fa-solid fa-house"></i> Về Trang chủ
              </Link>
            ) : (
              <Link to="/upload" className="btn primary">
                <i className="fa-solid fa-plus"></i> Upload Video Mới
              </Link>
            )}
            <button 
              onClick={onLogout} 
              className="btn logout-btn" 
              title="Đăng xuất"
              style={{ padding: '10px 12px' }}
            >
              <i className="fa-solid fa-right-from-bracket"></i>
            </button>
          </>
        ) : (
          <>
            <Link to="/auth" className="btn secondary" style={{ marginRight: '15px' }}>Đăng nhập</Link>
            <Link to="/upload" className="btn primary">Bắt đầu ngay</Link>
          </>
        )}
      </div>
    </header>
  );
};
