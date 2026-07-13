import React from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useToast } from '../context/ToastContext';
import { motion } from 'framer-motion';
import { 
  UploadCloud, 
  LayoutDashboard, 
  Settings, 
  LogOut, 
  Bell, 
  Globe 
} from 'lucide-react';

interface HeaderProps {
  isLoggedIn?: boolean;
  onLogout?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isLoggedIn = true, onLogout }) => {
  const toast = useToast();
  const location = useLocation();
  const { t, i18n } = useTranslation();
  const isAdminPage = location.pathname.startsWith('/admin');

  const toggleLanguage = () => {
    const newLang = i18n.language.startsWith('en') ? 'vi' : 'en';
    i18n.changeLanguage(newLang);
  };

  return (
    <header className="glass-panel flex justify-between items-center px-6 md:px-8 w-full h-16 sticky top-0 z-50">
      {/* Brand Logo */}
      <div className="flex items-center gap-3">
        <Link to={isAdminPage ? "/admin" : "/"} className="flex items-center gap-3 group">
          <motion.img 
            whileHover={{ rotate: 180 }}
            transition={{ duration: 0.5 }}
            alt="PrismVideo Logo" 
            className="w-8 h-8 object-contain" 
            src="/logo.svg" 
          />
          <span className="text-xl md:text-2xl font-heading font-bold text-slate-900 tracking-tight group-hover:text-primary transition-colors">
            {isAdminPage ? "Admin Panel" : "PrismVideo"}
          </span>
        </Link>
      </div>

      {/* Nav Links */}
      {!isAdminPage && (
        <nav className="hidden md:flex items-center gap-8 h-full">
          {[
            { path: '/', label: t('header.dashboard') },
            { path: '/history', label: t('header.history') },
            ...(isLoggedIn ? [{ path: '/profile', label: t('header.account') }] : []),
            { path: '/docs', label: t('header.docs') },
          ].map((navItem) => (
            <NavLink 
              key={navItem.path}
              to={navItem.path} 
              className={({ isActive }) => 
                `relative text-sm font-semibold transition-all duration-300 py-1 flex items-center ${
                  isActive ? 'text-primary' : 'text-slate-500 hover:text-slate-900'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {navItem.label}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute -bottom-1 left-0 right-0 h-[3px] bg-primary rounded-full"
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      )}

      {/* Action Buttons / User Info */}
      <div className="flex items-center gap-4">
        {isLoggedIn ? (
          <>
            {/* Quick Actions */}
            {!isAdminPage && (
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link 
                  to="/upload" 
                  className="btn primary !py-1.5 !px-4 !text-sm flex items-center gap-2 shadow-lg shadow-primary/20"
                >
                  <UploadCloud size={18} />
                  <span className="hidden sm:inline">{t('header.upload_video')}</span>
                </Link>
              </motion.div>
            )}

            {isAdminPage && (
              <Link 
                to="/" 
                className="btn secondary !py-1.5 !px-3 !text-sm flex items-center gap-2"
              >
                <LayoutDashboard size={18} />
                {t('header.dashboard')}
              </Link>
            )}

            <button onClick={toggleLanguage} className="flex items-center gap-1 font-bold text-xs uppercase px-2 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-100 transition-colors text-slate-700 cursor-pointer">
              <Globe size={14} />
              {i18n.language.startsWith('en') ? 'EN' : 'VI'}
            </button>

            <button className="text-slate-400 hover:text-primary p-2 rounded-full hover:bg-primary-light transition-all hidden sm:block cursor-pointer">
              <Bell size={20} />
            </button>
            
            <Link to="/admin" className="text-slate-400 hover:text-primary p-2 rounded-full hover:bg-primary-light transition-all hidden sm:block cursor-pointer" title="Admin Settings">
              <Settings size={20} />
            </Link>

            {/* Profile Avatar */}
            <motion.div whileHover={{ scale: 1.1 }}>
              <Link 
                to="/profile" 
                className="w-10 h-10 rounded-full overflow-hidden border-2 border-white shadow-md bg-slate-100 flex block"
              >
                <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                  <circle cx="50" cy="50" r="48" fill="#f8fafc" />
                  <path d="M50 55 C35 55 25 65 25 75 L25 80 L75 80 L75 75 C75 65 65 55 50 55 Z" fill="#6366f1"/>
                  <circle cx="50" cy="35" r="15" fill="#6366f1"/>
                  <circle cx="80" cy="80" r="8" fill="#10B981" stroke="#ffffff" strokeWidth="2"/>
                </svg>
              </Link>
            </motion.div>

            {/* Logout Button */}
            <button 
              onClick={() => {
                if (onLogout) onLogout();
                toast.success("Đăng xuất thành công!", "Thành công");
              }} 
              className="text-slate-400 hover:text-danger p-2 rounded-full hover:bg-danger-bg transition-all cursor-pointer"
              title="Đăng xuất"
            >
              <LogOut size={20} />
            </button>
          </>
        ) : (
          <>
            <button onClick={toggleLanguage} className="flex items-center gap-1 font-bold text-xs uppercase px-2 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-100 transition-colors text-slate-700 cursor-pointer">
              <Globe size={14} />
              {i18n.language.startsWith('en') ? 'EN' : 'VI'}
            </button>
            <Link 
              to="/auth" 
              className="btn secondary !py-1.5 !px-4 !text-sm"
            >
              {t('header.login')}
            </Link>
            <Link 
              to="/auth" 
              className="btn primary !py-1.5 !px-4 !text-sm shadow-lg shadow-primary/20"
            >
              {t('header.start_now')}
            </Link>
          </>
        )}
      </div>
    </header>
  );
};
