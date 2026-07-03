/**
 * StatsGrid — Grid hiển thị thống kê tổng hợp.
 * Chuyển đổi từ: UIs/5_admin.html (stats-grid)
 */
export default function StatsGrid() {
  const stats = [
    { label: 'Người dùng', value: '2,540', trend: '+12.5%', icon: 'fa-solid fa-users', iconColor: 'text-primary' },
    { label: 'Video Xử Lý', value: '8,932', trend: '+8.2%', icon: 'fa-solid fa-film', iconColor: 'text-primary' },
    { label: 'Audio WER', value: '7.8%', trend: 'Đạt chuẩn < 10%', icon: 'fa-solid fa-microphone-lines', iconColor: 'text-pink-500' },
    { label: 'Keyframe F-score', value: '0.52', trend: 'Vượt mục tiêu (0.45)', icon: 'fa-solid fa-image', iconColor: 'text-blue-500' },
  ];

  return (
    <div className="grid grid-cols-4 gap-5 mb-10">
      {stats.map((s, i) => (
        <div
          key={i}
          className="bg-white border border-gray-200 rounded-2xl p-6 transition-transform hover:-translate-y-1 hover:border-indigo-200 hover:shadow-lg shadow-sm"
        >
          <h3 className="text-gray-500 text-sm font-medium flex justify-between items-center mb-4">
            {s.label}
            <i className={`${s.icon} ${s.iconColor}`}></i>
          </h3>
          <div className="text-4xl font-outfit font-semibold">{s.value}</div>
          <span className="inline-block text-sm mt-3 px-2 py-1 rounded-full bg-emerald-50 text-emerald-600">
            <i className="fa-solid fa-arrow-trend-up mr-1"></i>{s.trend}
          </span>
        </div>
      ))}
    </div>
  );
}
