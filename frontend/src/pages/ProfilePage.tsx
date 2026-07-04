import React, { useState, useEffect } from "react";
import { api } from "../services/api";
import { UserRole } from "../types";
import "./ProfilePage.css";

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
      <div className="profile-page">
        <div style={{ textAlign: "center", marginTop: "50px" }}>
          <i
            className="fa-solid fa-spinner fa-spin"
            style={{ fontSize: "2rem", color: "#6366f1" }}
          ></i>
          <p style={{ marginTop: "10px", color: "#94a3b8" }}>
            Đang tải hồ sơ...
          </p>
        </div>
      </div>
    );
  }

  const initialLetter = profile?.email
    ? profile.email.charAt(0).toUpperCase()
    : "U";
  const isAdmin = profile?.role.toLowerCase() === UserRole.ADMIN;

  return (
    <div className="profile-page">
      <div className="container">
        {/* Profile Card Header */}
        <div className="profile-header-card">
          <div className="avatar-large">{initialLetter}</div>
          <div className="profile-title-area">
            <h2>{profile?.email || "Người dùng AI.Summarizer"}</h2>
            <span className={`role-badge ${isAdmin ? "admin" : "user"}`}>
              <i
                className={`fa-solid ${isAdmin ? "fa-user-shield" : "fa-user"}`}
              ></i>
              {isAdmin ? "Quản trị viên" : "Hội viên"}
            </span>
          </div>
        </div>

        {/* Profile Info Details Grid */}
        <div className="profile-grid">
          {/* Details Box */}
          <div className="profile-details-box">
            <h3>Chi tiết tài khoản</h3>
            <div className="detail-row">
              <span className="detail-label">Email</span>
              <span className="detail-value">{profile?.email}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Mã tài khoản (UUID)</span>
              <span className="detail-value">{profile?.userId}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Trạng thái</span>
              <span className="detail-value status-active">
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    backgroundColor: "#10b981",
                    display: "inline-block",
                  }}
                ></span>
                Đang hoạt động
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Vai trò hệ thống</span>
              <span className="detail-value">
                {isAdmin ? "Admin" : "User"}
              </span>
            </div>

            {/* Video Analysis Stats */}
            <div className="stats-counter">
              <i className="fa-solid fa-photo-film"></i>
              <div className="stats-info">
                <h4>{videoCount}</h4>
                <p>Bài giảng đã tải lên phân tích</p>
              </div>
            </div>
          </div>

          {/* Change Password / Settings Box */}
          <div className="profile-actions-box">
            <h3>Bảo mật & Cài đặt</h3>
            <form onSubmit={handlePasswordChange} className="action-form">
              <div className="form-group">
                <label>Mật khẩu cũ</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label>Mật khẩu mới</label>
                <input
                  type="password"
                  placeholder="Tối thiểu 6 ký tự"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label>Xác nhận mật khẩu mới</label>
                <input
                  type="password"
                  placeholder="Nhập lại mật khẩu mới"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>

              {passwordMsg && (
                <div
                  style={{
                    color: passwordMsg.type === "success" ? "#34d399" : "#f87171",
                    fontSize: "0.9rem",
                    fontWeight: "500",
                    marginTop: "5px",
                  }}
                >
                  <i
                    className={`fa-solid ${
                      passwordMsg.type === "success"
                        ? "fa-circle-check"
                        : "fa-circle-exclamation"
                    }`}
                    style={{ marginRight: "6px" }}
                  ></i>
                  {passwordMsg.text}
                </div>
              )}

              <button
                type="submit"
                className="btn primary"
                style={{ marginTop: "10px" }}
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
