/**
 * AdminSidebar — Sidebar navigation cho trang Admin.
 * Chuyển đổi từ: UIs/5_admin.html (sidebar)
 */
import { Link } from 'react-router-dom';

export default function AdminSidebar({ activeItem = 'reports' }) {
  const navGroups = [
    {
      label: 'Analytics',
      items: [
        { id: 'reports', icon: 'fa-solid fa-chart-pie', text: 'Báo cáo hệ thống' },
        { id: 'performance', icon: 'fa-solid fa-chart-line', text: 'Hiệu suất AI (WER/F-Score)' },
      ],
    },
    {
      label: 'Quản lý',
      items: [
        { id: 'users', icon: 'fa-solid fa-users', text: 'Tài khoản Người dùng' },
        { id: 'standards', icon: 'fa-solid fa-video', text: 'Tiêu chuẩn Video' },
        { id: 'queue', icon: 'fa-solid fa-server', text: 'Celery Job Queue' },
      ],
    },
  ];

  return (
    <div className="w-72 bg-white border-r border-gray-200 py-8 flex flex-col">
      <Link to="/admin" className="font-outfit text-xl font-semibold text-primary px-8 mb-10 flex items-center gap-3 no-underline">
        <i className="fa-solid fa-shield-halved"></i> Admin Panel
      </Link>

      {navGroups.map((group, gi) => (
        <div key={gi} className="mb-5">
          <div className="text-xs text-gray-500 uppercase tracking-wider px-8 mb-3 font-medium">
            {group.label}
          </div>
          {group.items.map((item) => (
            <a
              key={item.id}
              href="#"
              className={`flex items-center gap-4 px-8 py-3 text-sm no-underline border-l-[3px] transition-all
                ${activeItem === item.id
                  ? 'text-primary bg-indigo-50 border-l-primary'
                  : 'text-gray-500 border-l-transparent hover:text-primary hover:bg-indigo-50 hover:border-l-primary'}`}
            >
              <i className={item.icon}></i> {item.text}
            </a>
          ))}
        </div>
      ))}

      <div className="mt-auto">
        <Link to="/" className="flex items-center gap-4 px-8 py-3 text-sm text-gray-500 no-underline hover:text-primary hover:bg-indigo-50 border-l-[3px] border-l-transparent transition-all">
          <i className="fa-solid fa-house"></i> Về Trang chủ
        </Link>
        <Link to="/auth" className="flex items-center gap-4 px-8 py-3 text-sm text-gray-500 no-underline hover:text-primary hover:bg-indigo-50 border-l-[3px] border-l-transparent transition-all">
          <i className="fa-solid fa-right-from-bracket"></i> Đăng xuất
        </Link>
      </div>
    </div>
  );
}
