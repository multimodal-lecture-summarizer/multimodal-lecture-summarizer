import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, ArrowRight, ShieldCheck } from 'lucide-react';

interface AuthPageProps {
  onLogin?: (userData: { email: string; role: string }) => void;
}

export const AuthPage: React.FC<AuthPageProps> = ({ onLogin }) => {
  const toast = useToast();
  const { t } = useTranslation();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    
    try {
      const result = isLogin 
        ? await api.login(email, password)
        : await api.register(email, password);

      if (result.success && result.data) {
        let token = result.data.accessToken;
        
        if (!isLogin) {
          const loginRes = await api.login(email, password);
          token = loginRes.data.accessToken;
        }

        if (token) {
          localStorage.setItem('token', token);
          const profileResult = await api.getMe();
          if (profileResult.success && profileResult.data) {
            const userData = {
              email: profileResult.data.email,
              role: profileResult.data.role.toLowerCase(),
            };
            if (onLogin) onLogin(userData);
            toast.success(isLogin ? t('auth.login_success') : t('auth.register_success'), t('common.success'));
            
            if (userData.role === 'admin') {
              navigate('/admin');
            } else {
              navigate('/history');
            }
            return;
          }
        }
      }
      setErrorMessage(t('auth.no_token'));
      toast.error(t('auth.no_token'), t('common.error'));
    } catch (error: any) {
      console.error('API connection failed:', error);
      const errMsg = error.message || t('auth.conn_error');
      setErrorMessage(errMsg);
      toast.error(errMsg, t('common.error'));
    }
  };

  const mockSocialLogin = (provider: string) => {
    const role = email.trim().toLowerCase() === 'hungphitran.22@gmail.com' ? 'admin' : 'user';
    const defaultEmail = provider === 'google' ? 'google_researcher@gmail.com' : 'github_researcher@gmail.com';
    const userData = { email: email || defaultEmail, role: role };
    if (onLogin) onLogin(userData); 
    toast.success(provider === 'google' ? t('auth.google_success') : t('auth.github_success'), t('common.success'));
    if (role === 'admin') {
      navigate('/admin');
    } else {
      navigate('/history');
    }
  };

  return (
    <div className="min-h-[calc(100vh-64px)] w-full flex flex-col items-center justify-center px-4 py-12 relative overflow-hidden bg-[#FAF5FF]">
      
      {/* Animated Background Blobs */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <motion.div 
          animate={{ x: [0, 100, 0], y: [0, -50, 0] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="absolute -top-[20%] -left-[10%] w-[50vw] h-[50vw] rounded-full bg-primary/20 blur-[120px]" 
        />
        <motion.div 
          animate={{ x: [0, -100, 0], y: [0, 50, 0] }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          className="absolute top-[40%] -right-[10%] w-[40vw] h-[40vw] rounded-full bg-indigo-500/10 blur-[100px]" 
        />
      </div>

      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full flex justify-center pb-8 shrink-0"
      >
        <div className="flex flex-col items-center gap-1">
          <span className="font-heading text-3xl font-black text-slate-900 tracking-tight">PrismVideo</span>
          <span className="text-xs uppercase tracking-[0.3em] text-primary font-bold">{t('auth.portal')}</span>
        </div>
      </motion.header>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="w-full max-w-[480px] glass-panel rounded-[2rem] overflow-hidden flex flex-col shadow-xl shadow-primary/5 relative z-10"
      >
        {/* Tab switchers */}
        <div className="flex border-b border-slate-200/50 bg-white/40">
          <button 
            type="button"
            className="flex-1 py-5 text-sm font-bold relative transition-colors" 
            onClick={() => setIsLogin(true)}
          >
            <span className={`relative z-10 ${isLogin ? 'text-primary' : 'text-slate-500 hover:text-slate-700'}`}>
              {t('auth.login')}
            </span>
            {isLogin && <motion.div layoutId="activeAuthTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />}
          </button>
          <button 
            type="button"
            className="flex-1 py-5 text-sm font-bold relative transition-colors" 
            onClick={() => setIsLogin(false)}
          >
            <span className={`relative z-10 ${!isLogin ? 'text-primary' : 'text-slate-500 hover:text-slate-700'}`}>
              {t('auth.register')}
            </span>
            {!isLogin && <motion.div layoutId="activeAuthTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />}
          </button>
        </div>

        <div className="p-8 md:p-10 flex-grow bg-white/60">
          <AnimatePresence mode="wait">
            {errorMessage && (
              <motion.div 
                initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                animate={{ opacity: 1, height: 'auto', marginBottom: 16 }}
                exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                className="overflow-hidden"
              >
                <div className="p-4 bg-red-50 text-red-600 text-xs rounded-xl border border-red-200 font-bold flex items-center gap-2">
                  <ShieldCheck size={16} />
                  {errorMessage}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-5">
              <div className="flex flex-col gap-2">
                <label className="text-xs text-slate-700 font-bold ml-1">{t('auth.email_label')}</label>
                <div className="relative group">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-primary transition-colors" size={20} />
                  <input 
                    type="email" 
                    placeholder={t('auth.email_placeholder')} 
                    className="w-full pl-12 pr-4 py-3.5 bg-white border border-slate-200/60 rounded-2xl font-body text-sm placeholder:text-slate-400 focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-slate-900 outline-none shadow-sm"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required 
                  />
                </div>
              </div>
              
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center ml-1 mr-1">
                  <label className="text-xs text-slate-700 font-bold">{t('auth.password_label')}</label>
                  {isLogin && (
                    <a href="#forgot" className="text-[11px] text-primary hover:text-primary-hover font-bold transition-colors">{t('auth.forgot_password')}</a>
                  )}
                </div>
                <div className="relative group">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-primary transition-colors" size={20} />
                  <input 
                    type="password" 
                    placeholder={t('auth.password_placeholder')} 
                    className="w-full pl-12 pr-4 py-3.5 bg-white border border-slate-200/60 rounded-2xl font-body text-sm placeholder:text-slate-400 focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-slate-900 outline-none shadow-sm"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required 
                  />
                </div>
              </div>
            </div>
            
            <motion.button 
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              type="submit" 
              className="btn primary w-full py-4 text-sm tracking-wide shadow-lg shadow-primary/20 mt-2"
            >
              <span>{isLogin ? t('auth.btn_login') : t('auth.btn_register')}</span>
              <ArrowRight size={18} />
            </motion.button>
            
            <div className="relative py-4">
              <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-slate-200/60"></div></div>
              <div className="relative flex justify-center text-[10px] uppercase tracking-widest font-bold">
                <span className="bg-[#FAF5FF] px-4 text-slate-400">{t('auth.system_link')}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button 
                type="button"
                className="flex items-center justify-center gap-2.5 py-3.5 px-4 border border-slate-200/60 rounded-xl bg-white hover:bg-slate-50 hover:border-primary/50 transition-all shadow-sm font-bold text-xs text-slate-700"
                onClick={() => mockSocialLogin('google')}
              >
                <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
                  <path fill="#ea4335" d="M12 5.04c1.62 0 3.08.56 4.22 1.65l3.16-3.16C17.47 1.7 14.93 1 12 1 7.37 1 3.4 3.63 1.45 7.45l3.77 2.92C6.12 7.14 8.84 5.04 12 5.04z" />
                  <path fill="#4285f4" d="M23.49 12.27c0-.81-.07-1.59-.2-2.36H12v4.47h6.44c-.28 1.47-1.11 2.71-2.36 3.55l3.66 2.84c2.14-1.97 3.75-4.88 3.75-8.5z" />
                  <path fill="#fbbc05" d="M5.22 14.63c-.24-.71-.38-1.47-.38-2.26s.14-1.55.38-2.26L1.45 7.45C.52 9.27 0 11.29 0 13.41s.52 4.14 1.45 5.96l3.77-2.92c-.24-.71-.38-1.47-.38-2.26z" />
                  <path fill="#34a853" d="M12 23c3.24 0 5.97-1.07 7.96-2.92l-3.66-2.84c-1.01.68-2.31 1.08-3.8 1.08-3.16 0-5.88-2.1-6.78-5.33L.95 16.32C2.9 20.14 6.87 23 12 23z" />
                </svg>
                <span>{t('auth.google_login')}</span>
              </button>
              
              <button 
                type="button"
                className="flex items-center justify-center gap-2.5 py-3.5 px-4 border border-slate-200/60 rounded-xl bg-white hover:bg-slate-50 hover:border-primary/50 transition-all shadow-sm font-bold text-xs text-slate-700"
                onClick={() => mockSocialLogin('github')}
              >
                <svg className="w-4 h-4 shrink-0 text-slate-800" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
                </svg>
                <span>{t('auth.github_login')}</span>
              </button>
            </div>
          </form>
        </div>

        <div className="bg-emerald-50/50 px-8 py-4 flex items-center justify-center gap-2 border-t border-slate-200/50 shrink-0">
          <ShieldCheck className="text-emerald-500" size={16} />
          <span className="text-[11px] text-emerald-700 font-bold uppercase tracking-widest">{t('auth.encrypted_env')}</span>
        </div>
      </motion.div>
    </div>
  );
};
