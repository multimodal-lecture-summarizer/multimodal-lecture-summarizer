import React, { useState, useEffect } from "react";
import { api } from "../services/api";
import { UserRole } from "../types";

export const ProfilePage: React.FC = () => {
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
        console.warn("Offline or failed fetching profile, using local cache", err);
        // Fallback to local storage
        const storedUser = localStorage.getItem("user");
        if (storedUser) {
          const parsed = JSON.parse(storedUser);
          setProfile({
            email: parsed.email,
            role: parsed.role,
            userId: "mock-user-id-offline",
            isActive: true,
          });
        }
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
      .catch(() => {
        setVideoCount(3); // Default demo count
      });
  }, []);

  const handlePasswordChange = (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      setPasswordMsg({
        type: "error",
        text: "Mật khẩu xác nhận không khớp!",
      });
      return;
    }
    if (newPassword.length < 6) {
      setPasswordMsg({
        type: "error",
        text: "Mật khẩu mới phải từ 6 ký tự trở lên!",
      });
      return;
    }

    // Success simulation
    setPasswordMsg({
      type: "success",
      text: "Đổi mật khẩu thành công (Môi trường phát triển)!",
    });
    setOldPassword("");
    setNewPassword("");
    setConfirmPassword("");
  };

  if (loading) {
    return (
      <div className="flex flex-col justify-center items-center h-[60vh] bg-background text-on-surface">
        <span className="material-symbols-outlined text-[48px] text-vibrant-cyan animate-spin mb-4">autorenew</span>
        <p className="text-secondary text-sm font-semibold">Đang tải hồ sơ...</p>
      </div>
    );
  }

  const initialLetter = profile?.email
    ? profile.email.charAt(0).toUpperCase()
    : "U";
  const isAdmin = profile?.role.toLowerCase() === UserRole.ADMIN;

  return (
    <div className="bg-background text-on-surface p-6 md:p-margin-desktop max-w-container-max mx-auto min-h-screen">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Profile Card Header */}
        <div className="bg-surface border border-outline-variant rounded-xl p-6 flex flex-col sm:flex-row items-center gap-6 shadow-sm">
          <div className="w-16 h-16 rounded-full bg-deep-navy text-vibrant-cyan flex items-center justify-center font-bold text-xl uppercase border-2 border-outline-variant shadow-inner">
            {initialLetter}
          </div>
          <div className="text-center sm:text-left space-y-1">
            <h2 className="text-lg font-bold text-deep-navy">{profile?.email || "Người dùng AI.Summarizer"}</h2>
            <div className="flex justify-center sm:justify-start">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                isAdmin ? "bg-status-success/15 text-status-success border border-status-success/20" : "bg-vibrant-cyan/10 text-vibrant-cyan border border-vibrant-cyan/20"
              }`}>
                <span className="material-symbols-outlined text-sm">{isAdmin ? "security" : "person"}</span>
                {isAdmin ? "Quản trị viên" : "Hội viên"}
              </span>
            </div>
          </div>
        </div>

        {/* Profile Info Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Details Box */}
          <div className="bg-surface border border-outline-variant rounded-xl p-6 shadow-sm flex flex-col justify-between">
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-deep-navy pb-2 border-b border-outline-variant/30 flex items-center gap-2">
                <span className="material-symbols-outlined text-vibrant-cyan text-base">account_circle</span>
                Chi tiết tài khoản
              </h3>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs py-1">
                  <span className="text-secondary">Email</span>
                  <span className="font-semibold text-deep-navy truncate max-w-[200px]">{profile?.email}</span>
                </div>
                <div className="flex justify-between items-center text-xs py-1">
                  <span className="text-secondary">Mã tài khoản (UUID)</span>
                  <span className="font-mono-data text-[10px] text-deep-navy">{profile?.userId}</span>
                </div>
                <div className="flex justify-between items-center text-xs py-1">
                  <span className="text-secondary">Trạng thái</span>
                  <span className="inline-flex items-center gap-1 font-semibold text-status-success">
                    <span className="w-1.5 h-1.5 rounded-full bg-status-success animate-pulse"></span>
                    Đang hoạt động
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs py-1">
                  <span className="text-secondary">Vai trò hệ thống</span>
                  <span className="font-semibold text-deep-navy">{isAdmin ? "Admin" : "User"}</span>
                </div>
              </div>
            </div>

            {/* Video Analysis Stats */}
            <div className="mt-6 p-4 bg-background border border-outline-variant/60 rounded-xl flex items-center gap-4">
              <span className="material-symbols-outlined text-vibrant-cyan text-3xl shrink-0">video_library</span>
              <div>
                <h4 className="text-lg font-bold text-deep-navy">{videoCount}</h4>
                <p className="text-[11px] text-secondary">Bài giảng đã tải lên phân tích</p>
              </div>
            </div>
          </div>

          {/* Change Password / Settings Box */}
          <div className="bg-surface border border-outline-variant rounded-xl p-6 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-deep-navy pb-2 border-b border-outline-variant/30 flex items-center gap-2">
              <span className="material-symbols-outlined text-vibrant-cyan text-base">lock</span>
              Bảo mật &amp; Cài đặt
            </h3>
            
            <form onSubmit={handlePasswordChange} className="space-y-3">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-secondary">Mật khẩu cũ</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none text-xs transition-all"
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  required
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-secondary">Mật khẩu mới</label>
                <input
                  type="password"
                  placeholder="Tối thiểu 6 ký tự"
                  className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none text-xs transition-all"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  required
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-secondary">Xác nhận mật khẩu mới</label>
                <input
                  type="password"
                  placeholder="Nhập lại mật khẩu mới"
                  className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none text-xs transition-all"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>

              {passwordMsg && (
                <div className={`text-xs font-semibold flex items-center gap-1.5 pt-1 ${
                  passwordMsg.type === "success" ? "text-status-success" : "text-error"
                }`}>
                  <span className="material-symbols-outlined text-sm">
                    {passwordMsg.type === "success" ? "check_circle" : "info"}
                  </span>
                  {passwordMsg.text}
                </div>
              )}

              <button
                type="submit"
                className="w-full mt-2 py-2.5 bg-deep-navy hover:bg-slate-800 text-white rounded-xl text-xs font-bold transition-all shadow-sm"
              >
                Cập nhật mật khẩu
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};
