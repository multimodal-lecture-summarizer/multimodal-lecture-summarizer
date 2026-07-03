/**
 * AdminDashboard — Trang quản trị.
 * Trang: UIs/5_admin.html → frontend/src/pages/AdminDashboard.jsx
 */
import AdminSidebar from '../components/AdminSidebar';
import StatsGrid from '../components/StatsGrid';
import JobsTable from '../components/JobsTable';

export default function AdminDashboard() {
  return (
    <div className="flex min-h-screen bg-gray-50 overflow-x-hidden">
      <AdminSidebar activeItem="reports" />

      <div className="flex-1 p-10 overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-10">
          <h1 className="font-outfit text-3xl font-semibold m-0">Báo Cáo Thống Kê Tổng Hợp</h1>
          <div className="bg-white px-5 py-2.5 rounded-lg border border-gray-200 text-sm cursor-pointer shadow-sm">
            <i className="fa-regular fa-calendar mr-2"></i>Tháng 6, 2026
            <i className="fa-solid fa-chevron-down ml-3 text-xs"></i>
          </div>
        </div>

        <StatsGrid />

        {/* Charts placeholder */}
        <div className="grid grid-cols-[2fr_1fr] gap-6 mb-10">
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
            <h3 className="font-outfit text-lg mb-6 pb-4 border-b border-gray-200">Lưu lượng Video (Job Queue)</h3>
            <div className="h-48 bg-gray-50 rounded-lg flex items-center justify-center text-gray-400">
              <i className="fa-solid fa-chart-bar text-4xl mr-3"></i> Chart.js — Bar Chart
            </div>
          </div>
          <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
            <h3 className="font-outfit text-lg mb-6 pb-4 border-b border-gray-200">Sử dụng Mô hình LLM</h3>
            <div className="h-48 bg-gray-50 rounded-lg flex items-center justify-center text-gray-400">
              <i className="fa-solid fa-chart-pie text-4xl mr-3"></i> Chart.js — Doughnut
            </div>
          </div>
        </div>

        <JobsTable />
      </div>
    </div>
  );
}
