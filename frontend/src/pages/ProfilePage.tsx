import React, { useState, useEffect } from "react";
import { api } from "../services/api";
import { UserRole } from "../types";
import { useToast } from "../context/ToastContext";
import { Skeleton } from "../components/Skeleton";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, Shield, Lock, Video, CheckCircle2, 
  AlertCircle, KeyRound, Mail, Settings 
} from 'lucide-react';

export const ProfilePage: React.FC = () => {
  const toast = useToast();
  const { t } = useTranslation();
  const [profile, setProfile] = useState<{
    email: string;
    role: string;
    userId: string;
    isActive: boolean;
  } | null>(null);
  const [videoCount, setVideoCount] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  // Password state for simulation
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordMsg, setPasswordMsg] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  useEffect(() => {
    // 1. Fetch user profile
    api
      .getMe()
      .then((res) => {
        if (res.success && res.data) {
          setProfile({
            email: res.data.email,
            role: res.data.role,
            userId: res.data.userId,
            isActive: res.data.isActive,
          });
        }
      })
      .catch((err) => {
        console.error("Failed fetching profile from backend API", err);
        setProfile(null);
      })
      .finally(() => {
        setLoading(false);
      });

    // 2. Fetch videos to get the user's upload count
    api
      .getVideos()
      .then((res) => {
        if (res.success && res.data) {
          setVideoCount(res.data.length);
        }
      })
      .catch((err) => {
        console.error("Failed to load user video list count", err);
        setVideoCount(0);
      });
  }, []);

  const handlePasswordChange = (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      setPasswordMsg({
        type: "error",
        text: t("profile.pass_mismatch"),
      });
      toast.error(t("profile.pass_mismatch"), t("common.error"));
      return;
    }
    if (newPassword.length < 6) {
      setPasswordMsg({
        type: "error",
        text: t("profile.pass_length"),
      });
      toast.error(t("profile.pass_length"), t("common.error"));
      return;
    }

    // Success simulation
    setPasswordMsg({
      type: "success",
      text: t("profile.pass_success"),
    });
    toast.success(t("profile.pass_success"), t("common.success"));
    setOldPassword("");
    setNewPassword("");
    setConfirmPassword("");
  };

  if (loading) {
    return (
      <div className="bg-[#FAF5FF] min-h-[calc(100vh-64px)] p-6 md:p-12 w-full flex justify-center">
        <div className="max-w-4xl w-full space-y-6">
          <Skeleton.Card className="rounded-[2rem] border-none shadow-sm h-[140px]" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Skeleton.Card className="rounded-[2rem] border-none shadow-sm h-[300px]" />
            <Skeleton.Card className="rounded-[2rem] border-none shadow-sm h-[300px]" />
          </div>
        </div>
      </div>
    );
  }

  const initialLetter = profile?.email
    ? profile.email.charAt(0).toUpperCase()
    : "U";
  const isAdmin = profile?.role.toLowerCase() === UserRole.ADMIN;

  return (
    <div className="bg-[#FAF5FF] min-h-[calc(100vh-64px)] p-6 md:p-12 w-full flex justify-center relative overflow-hidden">
      
      {/* Background Blobs */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-primary/5 rounded-full blur-[100px] pointer-events-none -z-10" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-indigo-500/5 rounded-full blur-[100px] pointer-events-none -z-10" />

      <div className="max-w-4xl w-full space-y-6 relative z-10">
        {/* Profile Card Header */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="glass-panel rounded-[2rem] p-8 flex flex-col sm:flex-row items-center gap-8 shadow-xl shadow-primary/5 border border-white/60"
        >
          <div className="relative">
            <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-primary to-indigo-600 text-white flex items-center justify-center font-black text-4xl shadow-lg shadow-primary/30 border-4 border-white">
              {initialLetter}
            </div>
            {profile?.isActive && (
              <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-emerald-500 rounded-full border-4 border-white flex items-center justify-center text-white" title="Active">
                <CheckCircle2 size={16} />
              </div>
            )}
          </div>
          
          <div className="text-center sm:text-left space-y-3">
            <h2 className="font-heading text-3xl font-black text-slate-900 tracking-tight">
              {profile?.email || t("profile.default_name")}
            </h2>
            <div className="flex justify-center sm:justify-start gap-3">
              <span className={`inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-bold shadow-sm ${
                isAdmin 
                  ? "bg-gradient-to-r from-emerald-100 to-emerald-50 text-emerald-700 border border-emerald-200/50" 
                  : "bg-gradient-to-r from-slate-100 to-white text-slate-700 border border-slate-200/50"
              }`}>
                {isAdmin ? <Shield size={14} /> : <User size={14} />}
                {isAdmin ? t("profile.role_admin") : t("profile.role_user")}
              </span>
            </div>
          </div>
        </motion.div>

        {/* Profile Info Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Details Box */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}
            className="glass-panel rounded-[2rem] p-8 shadow-sm flex flex-col justify-between border border-white/60"
          >
            <div className="space-y-6">
              <h3 className="font-heading text-lg font-bold text-slate-900 flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                  <User size={18} />
                </div>
                {t("profile.account_details")}
              </h3>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center bg-white/50 px-4 py-3 rounded-xl border border-slate-100">
                  <span className="text-sm text-slate-500 flex items-center gap-2">
                    <Mail size={16} className="text-slate-400" />
                    {t("profile.email")}
                  </span>
                  <span className="font-bold text-slate-900 truncate max-w-[200px]">{profile?.email}</span>
                </div>
                
                <div className="flex justify-between items-center bg-white/50 px-4 py-3 rounded-xl border border-slate-100">
                  <span className="text-sm text-slate-500 flex items-center gap-2">
                    <KeyRound size={16} className="text-slate-400" />
                    {t("profile.uuid")}
                  </span>
                  <span className="font-mono text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-md">{profile?.userId}</span>
                </div>
                
                <div className="flex justify-between items-center bg-white/50 px-4 py-3 rounded-xl border border-slate-100">
                  <span className="text-sm text-slate-500 flex items-center gap-2">
                    <Settings size={16} className="text-slate-400" />
                    {t("profile.system_role")}
                  </span>
                  <span className={`font-bold ${isAdmin ? 'text-emerald-600' : 'text-primary'}`}>
                    {isAdmin ? t("profile.admin") : t("profile.user")}
                  </span>
                </div>
              </div>
            </div>

            {/* Video Analysis Stats */}
            <motion.div 
              whileHover={{ scale: 1.02 }}
              className="mt-8 p-6 bg-primary/5 rounded-[1.5rem] flex items-center gap-5 border border-primary/20"
            >
              <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center shrink-0">
                <Video size={24} className="text-primary" />
              </div>
              <div>
                <h4 className="font-heading text-3xl font-black text-primary">{videoCount}</h4>
                <p className="text-sm text-slate-600 font-bold">{t("profile.videos_analyzed")}</p>
              </div>
            </motion.div>
          </motion.div>

          {/* Change Password Box */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}
            className="glass-panel rounded-[2rem] p-8 shadow-sm border border-white/60"
          >
            <h3 className="font-heading text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-orange-100 text-orange-500 flex items-center justify-center">
                <Lock size={18} />
              </div>
              {t("profile.security_settings")}
            </h3>
            
            <form onSubmit={handlePasswordChange} className="space-y-4">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-bold text-slate-600 ml-1">{t("profile.old_pass")}</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                  <input
                    type="password"
                    placeholder="••••••••"
                    className="w-full pl-11 pr-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm transition-all shadow-sm"
                    value={oldPassword}
                    onChange={(e) => setOldPassword(e.target.value)}
                    required
                  />
                </div>
              </div>
              
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-bold text-slate-600 ml-1">{t("profile.new_pass")}</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                  <input
                    type="password"
                    placeholder={t("profile.new_pass_placeholder")}
                    className="w-full pl-11 pr-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm transition-all shadow-sm"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                  />
                </div>
              </div>
              
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-bold text-slate-600 ml-1">{t("profile.confirm_pass")}</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                  <input
                    type="password"
                    placeholder={t("profile.confirm_pass_placeholder")}
                    className="w-full pl-11 pr-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm transition-all shadow-sm"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              <AnimatePresence>
                {passwordMsg && (
                  <motion.div 
                    initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                    className={`text-xs font-bold flex items-center gap-2 p-3 rounded-xl border ${
                      passwordMsg.type === "success" 
                        ? "bg-emerald-50 text-emerald-600 border-emerald-100" 
                        : "bg-red-50 text-red-500 border-red-100"
                    }`}
                  >
                    {passwordMsg.type === "success" ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                    {passwordMsg.text}
                  </motion.div>
                )}
              </AnimatePresence>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="btn primary w-full mt-4 py-3 shadow-lg shadow-primary/20"
              >
                {t("profile.update_pass")}
              </motion.button>
            </form>
          </motion.div>
        </div>
      </div>
    </div>
  );
};
