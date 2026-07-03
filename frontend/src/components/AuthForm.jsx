/**
 * AuthForm — Form đăng nhập / đăng ký.
 * Chuyển đổi từ: UIs/7_auth.html
 */
import { Link } from 'react-router-dom';

export default function AuthForm() {
  return (
    <div className="bg-white p-10 rounded-2xl border border-gray-200 shadow-lg w-full max-w-md text-center">
      <Link to="/" className="text-2xl font-outfit font-bold text-primary no-underline inline-flex items-center gap-3 mb-8">
        <i className="fa-solid fa-brain"></i> AI.Summarizer
      </Link>
      <h2 className="font-outfit text-xl font-semibold mb-2">Chào mừng trở lại</h2>
      <p className="text-gray-500 text-sm mb-8">Đăng nhập để xem lịch sử và phân tích video</p>

      <form action="/history" className="text-left">
        <div className="mb-5">
          <label className="block text-sm font-medium mb-2">Email</label>
          <input
            type="email"
            placeholder="Nhập địa chỉ email"
            required
            className="w-full px-4 py-3 rounded-lg border border-gray-200 outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
          />
        </div>
        <div className="mb-5">
          <label className="block text-sm font-medium mb-2">Mật khẩu</label>
          <input
            type="password"
            placeholder="Nhập mật khẩu"
            required
            className="w-full px-4 py-3 rounded-lg border border-gray-200 outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
          />
        </div>

        <div className="flex justify-between mb-6 text-sm">
          <label className="flex items-center gap-2 cursor-pointer text-gray-500">
            <input type="checkbox" /> Ghi nhớ tôi
          </label>
          <a href="#" className="text-primary no-underline hover:underline">Quên mật khẩu?</a>
        </div>

        <button type="submit" className="w-full bg-primary text-white py-3 rounded-lg font-semibold text-base cursor-pointer hover:bg-primary-hover transition-all">
          Đăng Nhập
        </button>
      </form>

      {/* Divider */}
      <div className="flex items-center my-6 text-gray-500 text-sm">
        <span className="flex-1 h-px bg-gray-200"></span>
        <span className="px-4">HOẶC</span>
        <span className="flex-1 h-px bg-gray-200"></span>
      </div>

      <button className="w-full bg-white border border-gray-200 text-gray-900 py-3 rounded-lg font-semibold text-base cursor-pointer hover:bg-gray-50 hover:border-gray-300 transition-all flex justify-center items-center gap-3">
        <i className="fa-brands fa-google text-red-500"></i> Đăng nhập bằng Google
      </button>

      <p className="mt-6 text-sm text-gray-500">
        Chưa có tài khoản? <a href="#" className="text-primary no-underline font-medium hover:underline">Đăng ký ngay</a>
      </p>
    </div>
  );
}
