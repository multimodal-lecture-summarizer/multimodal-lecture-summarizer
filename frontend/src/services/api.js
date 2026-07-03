/**
 * API Client — Gọi Backend FastAPI.
 * Sử dụng axios để giao tiếp với backend_api.
 */
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor — attach JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor — handle 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/auth';
    }
    return Promise.reject(error);
  }
);

// --- Video API ---
export const uploadVideo = (formData, onProgress) =>
  api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  });

export const getVideoStatus = (videoId) => api.get(`/videos/${videoId}/status`);
export const getVideoResults = (videoId) => api.get(`/videos/${videoId}/results`);
export const getVideoHistory = () => api.get('/videos');
export const deleteVideo = (videoId) => api.delete(`/videos/${videoId}`);

// --- Q&A API ---
export const askQuestion = (videoId, question) =>
  api.post(`/videos/${videoId}/qa`, { question });

// --- Auth API ---
export const login = (email, password) => api.post('/auth/login', { email, password });
export const register = (email, password) => api.post('/auth/register', { email, password });

// --- Admin API ---
export const getStats = () => api.get('/stats');
export const getJobs = () => api.get('/admin/jobs');

export default api;
