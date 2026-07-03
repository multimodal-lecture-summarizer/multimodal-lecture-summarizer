/**
 * HistoryCard — Card hiển thị video đã xử lý.
 * Chuyển đổi từ: UIs/6_history.html
 */
import { Link } from 'react-router-dom';

export default function HistoryCard({ video }) {
  const statusClasses = {
    done: 'bg-emerald-50 text-emerald-700',
    processing: 'bg-indigo-50 text-indigo-700',
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 flex items-center gap-5 shadow-sm hover:-translate-y-1 hover:shadow-lg hover:border-indigo-200 transition-all mb-5">
      {/* Thumbnail */}
      <div className="w-40 h-24 bg-gray-200 rounded-lg flex justify-center items-center text-gray-500 shrink-0 overflow-hidden">
        {video.thumbnailIcon ? (
          <i className={`${video.thumbnailIcon} text-3xl`}></i>
        ) : (
          <i className="fa-solid fa-image text-3xl"></i>
        )}
      </div>

      {/* Info */}
      <div className="flex-1">
        <h3 className="text-base font-semibold mb-2">{video.title}</h3>
        <div className="flex gap-5 text-gray-500 text-sm mb-2">
          <span className="flex items-center gap-1"><i className="fa-regular fa-clock"></i> {video.duration}</span>
          <span className="flex items-center gap-1"><i className="fa-regular fa-calendar"></i> {video.date}</span>
        </div>
        <span className={`inline-block px-2.5 py-1 rounded-md text-xs font-semibold ${statusClasses[video.status]}`}>
          {video.status === 'done' ? (
            <><i className="fa-solid fa-check mr-1"></i>Hoàn tất</>
          ) : (
            <><i className="fa-solid fa-spinner fa-spin mr-1"></i>{video.statusText || 'Đang xử lý...'}</>
          )}
        </span>
      </div>

      {/* Actions */}
      <div className="flex gap-3 shrink-0">
        {video.status === 'done' ? (
          <Link to="/results" className="bg-primary text-white px-4 py-2 rounded-md text-sm font-medium no-underline hover:bg-primary-hover transition-all flex items-center gap-2">
            <i className="fa-regular fa-eye"></i> Xem Kết Quả
          </Link>
        ) : (
          <button disabled className="bg-white border border-gray-200 text-gray-500 px-4 py-2 rounded-md text-sm opacity-50 cursor-not-allowed flex items-center gap-2">
            <i className="fa-regular fa-eye"></i> Đang Xử Lý
          </button>
        )}
        <button className="bg-white border border-red-200 text-red-500 px-3 py-2 rounded-md text-sm hover:bg-red-50 transition-all">
          <i className="fa-regular fa-trash-can"></i>
        </button>
      </div>
    </div>
  );
}
