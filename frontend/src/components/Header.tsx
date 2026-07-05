import React from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';

interface HeaderProps {
  isLoggedIn?: boolean;
  onLogout?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isLoggedIn = true, onLogout }) => {
  const location = useLocation();
  const isAdminPage = location.pathname.startsWith('/admin');

  return (
    <header className="bg-surface-container-lowest flex justify-between items-center px-6 md:px-margin-desktop w-full h-16 sticky top-0 z-50 border-b border-outline-variant">
      {/* Brand Logo */}
      <div className="flex items-center gap-3">
        <Link to={isAdminPage ? "/admin" : "/"} className="flex items-center gap-3">
          <img 
            alt="Multimodal Lecture Summarizer Logo" 
            className="w-8 h-8 object-contain" 
            src="https://lh3.googleusercontent.com/aida/AP1WRLtxduNxaNOmKabESZ0gKYWnEo9KvTKhU0LUYJGKGTYUY23EXRtO-soFY7IIoUor2nEuaKvZ40vnz86z_Ks7NI2pra0JvnlX5bSe2gcR5SCmCPQ390eJNg5ROT7jq4kzkBUQoYGdsBjBsyVko-OacbCQ2vOIuF-eUvVV0yV2jzFXWQVnAY2HvQ8wbLk8pu3Zqbb-UbtA7Ep2tDEtS4avM7r7XW0klIOWYv1eAa27KEW7gzQ0bY7WnhCcWV8"
          />
          <span className="font-headline-lg text-xl md:text-headline-lg font-bold text-deep-navy tracking-tight">
            {isAdminPage ? "Admin Panel" : "Multimodal Lecture Summarizer"}
          </span>
        </Link>
      </div>

      {/* Nav Links */}
      {!isAdminPage && (
        <nav className="hidden md:flex items-center gap-8 h-full">
          <NavLink 
            to="/" 
            className={({ isActive }) => 
              `font-label-md text-label-md transition-all duration-150 pb-1 ${
                isActive 
                  ? 'text-primary border-b-2 border-primary' 
                  : 'text-secondary hover:text-primary'
              }`
            }
          >
            Dashboard
          </NavLink>
          <NavLink 
            to="/history" 
            className={({ isActive }) => 
              `font-label-md text-label-md transition-all duration-150 pb-1 ${
                isActive 
                  ? 'text-primary border-b-2 border-primary' 
                  : 'text-secondary hover:text-primary'
              }`
            }
          >
            Lịch sử video
          </NavLink>
          {isLoggedIn && (
            <NavLink 
              to="/profile" 
              className={({ isActive }) => 
                `font-label-md text-label-md transition-all duration-150 pb-1 ${
                  isActive 
                    ? 'text-primary border-b-2 border-primary' 
                    : 'text-secondary hover:text-primary'
                }`
              }
            >
              Tài khoản
            </NavLink>
          )}
          <NavLink 
            to="/docs" 
            className={({ isActive }) => 
              `font-label-md text-label-md transition-all duration-150 pb-1 ${
                isActive 
                  ? 'text-primary border-b-2 border-primary' 
                  : 'text-secondary hover:text-primary'
              }`
            }
          >
            Tài liệu
          </NavLink>
        </nav>
      )}

      {/* Action Buttons / User Info */}
      <div className="flex items-center gap-4">
        {isLoggedIn ? (
          <>
            {/* Quick Actions */}
            {!isAdminPage && (
              <Link 
                to="/upload" 
                className="px-4 py-2 bg-deep-navy text-on-primary font-label-md text-label-sm rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-all scale-active-95"
              >
                <span className="material-symbols-outlined text-[18px]">add</span>
                <span className="hidden sm:inline">Upload Video</span>
              </Link>
            )}

            {isAdminPage && (
              <Link 
                to="/" 
                className="px-4 py-2 border border-slate-600 text-slate-600 font-label-md text-label-sm rounded-lg flex items-center justify-center gap-2 hover:bg-slate-100 transition-all"
              >
                <span className="material-symbols-outlined text-[18px]">home</span>
                Dashboard
              </Link>
            )}

            <button className="text-secondary hover:text-primary p-1.5 transition-colors hidden sm:block">
              <span className="material-symbols-outlined">notifications</span>
            </button>
            
            <Link to="/admin" className="text-secondary hover:text-primary p-1.5 transition-colors hidden sm:block" title="Admin Settings">
              <span className="material-symbols-outlined">settings</span>
            </Link>

            {/* Profile Avatar */}
            <Link 
              to="/profile" 
              className="w-10 h-10 rounded-full overflow-hidden border border-outline-variant bg-surface-container-high transition-transform hover:scale-105"
            >
              <img 
                className="w-full h-full object-cover" 
                alt="Profile Avatar" 
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCv51dKL0AznsnRknP48ChToixvy8nIBmicv8oKCXJpiOkdOI6ziho0kAdHrGNj8DkXfwZz7VDLkLB_1YtJ8ILnH8WkQQpeS64p46p9XY0iAUEX1hMcDQMgJqS9PZFFSJ-MrOg1XK0CyCHzlw4lsokBZpacc6cPoCpEC4_SgadGsMunGYgeQ3EW6Z-7RSuk_pqMhvc8u0YHV5IwJcWQQVwK9HyFV97KzG_j-RxCouKJejMm_MLI51Wu"
              />
            </Link>

            {/* Logout Button */}
            <button 
              onClick={onLogout} 
              className="text-secondary hover:text-error p-1.5 transition-colors"
              title="Đăng xuất"
            >
              <span className="material-symbols-outlined">logout</span>
            </button>
          </>
        ) : (
          <>
            <Link 
              to="/auth" 
              className="px-4 py-2 border border-slate-600 text-slate-600 font-label-md text-label-sm rounded-lg flex items-center justify-center gap-2 hover:bg-slate-100 transition-all"
            >
              Đăng nhập
            </Link>
            <Link 
              to="/auth" 
              className="px-4 py-2 bg-deep-navy text-on-primary font-label-md text-label-sm rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-all scale-active-95"
            >
              Bắt đầu ngay
            </Link>
          </>
        )}
      </div>
    </header>
  );
};
