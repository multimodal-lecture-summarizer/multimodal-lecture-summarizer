import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './AuthPage.css';

interface AuthPageProps {
  onLogin?: () => void;
}

export const AuthPage: React.FC<AuthPageProps> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onLogin) onLogin();
    navigate('/history');
  };

  return (
    <div className="auth-page animate-fade-in">
      <div className="auth-container">
        <Link to="/" className="logo">
          <i className="fa-solid fa-brain"></i> AI.Summarizer
        </Link>
        
        <h2>{isLogin ? 'Chào mừng trở lại' : 'Tạo tài khoản mới'}</h2>
        <p>{isLogin ? 'Đăng nhập để xem lịch sử và phân tích video' : 'Đăng ký để lưu trữ và quản lý các video bài giảng'}</p>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Email</label>
            <input 
              type="email" 
              placeholder="Nhập địa chỉ email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required 
            />
          </div>
          
          <div className="form-group">
            <label>Mật khẩu</label>
            <input 
              type="password" 
              placeholder="Nhập mật khẩu" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required 
            />
          </div>
          
          {isLogin ? (
            <div className="form-options">
              <label className="remember-me">
                <input type="checkbox" /> Ghi nhớ tôi
              </label>
              <a href="#forgot" className="forgot-link">Quên mật khẩu?</a>
            </div>
          ) : null}
          
          <button type="submit" className="btn primary submit-btn">
            {isLogin ? 'Đăng Nhập' : 'Đăng Ký'}
          </button>
        </form>
        
        <div className="divider"><span>HOẶC</span></div>
        
        <button className="btn btn-google" onClick={() => { if (onLogin) onLogin(); navigate('/history'); }}>
          <i className="fa-brands fa-google" style={{ color: '#ea4335' }}></i> Đăng nhập bằng Google
        </button>
        
        <div className="footer-text">
          {isLogin ? (
            <>
              Chưa có tài khoản?{' '}
              <a href="#register" onClick={() => setIsLogin(false)}>Đăng ký ngay</a>
            </>
          ) : (
            <>
              Đã có tài khoản?{' '}
              <a href="#login" onClick={() => setIsLogin(true)}>Đăng nhập ngay</a>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
