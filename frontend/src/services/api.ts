import { CONFIG } from "../config";

const getAuthHeaders = (isMultipart = false) => {
  const token = localStorage.getItem("token");
  return {
    ...(isMultipart ? {} : { "Content-Type": "application/json" }),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

const customFetch = async (url: string, options: RequestInit = {}) => {
  const response = await fetch(url, options);
  if (response.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    if (
      typeof window !== "undefined" &&
      !window.location.hash.includes("/auth")
    ) {
      window.location.hash = "#/auth";
      window.location.reload();
    }
  }
  return response;
};

export const api = {
  // Auth endpoints
  async login(email: string, password: string) {
    const response = await customFetch(`${CONFIG.API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ email, password }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Đăng nhập thất bại");
    return result; // returns BaseDTO[Token]
  },

  async register(email: string, password: string) {
    const response = await customFetch(`${CONFIG.API_BASE_URL}/auth/register`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ email, password }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Đăng ký thất bại");
    return result; // returns BaseDTO[UserDTO]
  },

  async getMe() {
    const response = await customFetch(`${CONFIG.API_BASE_URL}/auth/me`, {
      headers: getAuthHeaders(),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy thông tin cá nhân");
    return result; // returns BaseDTO[UserDTO]
  },

  async forgotPassword(email: string) {
    const response = await customFetch(`${CONFIG.API_BASE_URL}/auth/forgot-password`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ email }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Yêu cầu khôi phục thất bại");
    return result;
  },

  async resetPassword(email: string, newPassword: string) {
    const response = await customFetch(`${CONFIG.API_BASE_URL}/auth/reset-password`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ email, newPassword }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Đặt lại mật khẩu thất bại");
    return result;
  },


  // Videos endpoints
  async uploadVideo(
    file: File | null,
    originalUrl: string | null,
    language = "en"
  ) {
    const formData = new FormData();
    if (file) formData.append("file", file);
    if (originalUrl) formData.append("originalUrl", originalUrl);
    formData.append("language", language);

    const response = await customFetch(`${CONFIG.API_BASE_URL}/videos/upload`, {
      method: "POST",
      headers: getAuthHeaders(true),
      body: formData,
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Tải video lên thất bại");
    return result; // returns BaseDTO[VideoDTO]
  },

  async getVideos(status?: string, limit = 20, offset = 0, searchQuery?: string, sortBy = "newest") {
    let url = `${CONFIG.API_BASE_URL}/videos?limit=${limit}&offset=${offset}&sort_by=${sortBy}`;
    if (status) url += `&status=${status}`;
    if (searchQuery) url += `&search_query=${encodeURIComponent(searchQuery)}`;

    const response = await customFetch(url, {
      headers: getAuthHeaders(),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy danh sách video");
    return result; // returns BaseDTO[List[VideoDTO]]
  },

  async getVideo(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/${videoId}`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy chi tiết video");
    return result; // returns BaseDTO[VideoDTO]
  },

  async getVideoScenes(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/${videoId}/scenes`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy danh sách phân đoạn video");
    return result; // returns BaseDTO[List[VideoSceneDTO]]
  },

  // Jobs endpoints
  async getJobStatus(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/jobs/video/${videoId}`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy trạng thái tiến trình");
    return result; // returns BaseDTO[JobDTO] or similar
  },

  // Summaries endpoints
  async getSummary(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/summaries/video/${videoId}`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy tóm tắt video");
    return result; // returns BaseDTO[SummaryDTO]
  },

  async exportSummary(videoId: string, format: "txt" | "srt" | "pdf") {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/summaries/video/${videoId}/export?format=${format}`,
      {
        headers: getAuthHeaders(),
      }
    );
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      throw new Error(result.message || "Xuất tóm tắt thất bại");
    }
    return await response.blob();
  },

  // QA endpoints
  async askQuestion(videoId: string, question: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/qa/video/${videoId}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({ question }),
      }
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Hỏi đáp thất bại");
    return result; // returns BaseDTO[QAResponseDTO]
  },

  async getQaHistory(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/qa/video/${videoId}/history`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Lấy lịch sử chat thất bại");
    return result; // returns BaseDTO[List[QALogDTO]]
  },

  async clearQaHistory(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/qa/video/${videoId}/history`,
      {
        method: "DELETE",
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "Xóa lịch sử chat thất bại");
    return result;
  },


  // Video standards config
  async getStandards() {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/standards`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        result.message || "Không thể lấy cấu hình tiêu chuẩn video"
      );
    return result; // returns BaseDTO[VideoStandardDTO]
  },

  async updateStandards(standards: {
    maxDuration: number;
    allowedFormats: string;
    maxFileSize: number;
    minAudioQuality: number;
  }) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/standards`,
      {
        method: "PUT",
        headers: getAuthHeaders(),
        body: JSON.stringify(standards),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        result.message || "Không thể cập nhật cấu hình tiêu chuẩn video"
      );
    return result; // returns BaseDTO[VideoStandardDTO]
  },

  // Users Management
  async getUsers(limit = 10, offset = 0) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/users?limit=${limit}&offset=${offset}`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy danh sách người dùng");
    return result; // returns BaseDTO[List[UserDTO]]
  },

  async toggleUserStatus(userId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/users/${userId}/status`,
      {
        method: "PUT",
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể đổi trạng thái người dùng");
    return result; // returns BaseDTO[UserDTO]
  },

  async changeUserRole(userId: string, role: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/users/${userId}/role?role=${role}`,
      {
        method: "PUT",
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể thay đổi quyền người dùng");
    return result; // returns BaseDTO[UserDTO]
  },

  // Admin video & job list management
  async getAllVideosAdmin(limit = 50, offset = 0) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/admin/all?limit=${limit}&offset=${offset}`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể lấy toàn bộ video hệ thống");
    return result; // returns BaseDTO[List[VideoDTO]]
  },

  async deleteVideo(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/${videoId}`,
      {
        method: "DELETE",
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể xóa video của bạn");
    return result; // returns BaseDTO[bool]
  },

  async deleteVideoAdmin(videoId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/videos/admin/${videoId}`,
      {
        method: "DELETE",
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể xóa video hệ thống");
    return result; // returns BaseDTO[bool]
  },

  async getAllJobsAdmin(limit = 50, offset = 0) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/jobs/admin/all?limit=${limit}&offset=${offset}`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        result.message || "Không thể lấy toàn bộ tiến trình hệ thống"
      );
    return result; // returns BaseDTO[List[JobDTO]]
  },

  // Admin stats endpoint
  async getAdminStats() {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/stats/dashboard`,
      {
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        result.message || "Không thể lấy báo cáo thống kê quản trị"
      );
    return result; // returns BaseDTO[AdminStatsDTO]
  },

  async cancelJob(jobId: string) {
    const response = await customFetch(
      `${CONFIG.API_BASE_URL}/jobs/${jobId}/cancel`,
      {
        method: "POST",
        headers: getAuthHeaders(),
      }
    );
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.message || "Không thể dừng tiến trình này");
    return result; // returns BaseDTO[JobDTO]
  },
};
