/**
 * JobsTable — Bảng tác vụ xử lý gần đây.
 * Chuyển đổi từ: UIs/5_admin.html (table)
 */
export default function JobsTable() {
  const jobs = [
    { id: '#UUID-A8F9', email: 'nguyen.a@gmail.com', name: 'MIT 6.S191 - Deep Learning', duration: '45:10', status: 'done', statusText: 'Hoàn tất (2m 15s)', pipeline: 'WhisperX + GPT-4o' },
    { id: '#UUID-B2E1', email: 'tran.b@student.edu.vn', name: 'TED Talk - Future of AI', duration: '12:45', status: 'processing', statusText: 'Visual Module (CLIP)', pipeline: 'WhisperX + Gemini' },
    { id: '#UUID-C4D5', email: 'le.c@company.com', name: 'Internal Training 2026.mp4', duration: '1:20:00', status: 'failed', statusText: 'Lỗi: Vượt quá 60 phút', pipeline: '-' },
    { id: '#UUID-D7F2', email: 'admin@system.local', name: 'Stanford CS224N - NLP', duration: '55:30', status: 'done', statusText: 'Hoàn tất (3m 05s)', pipeline: 'WhisperX + Qwen2.5 Local' },
  ];

  const statusClasses = {
    done: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
    processing: 'bg-indigo-50 text-indigo-700 border border-indigo-200',
    failed: 'bg-red-50 text-red-700 border border-red-200',
  };

  return (
    <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
      <h3 className="font-outfit text-lg p-6 m-0 border-b border-gray-200">
        Tác vụ Xử lý Gần Đây (PostgreSQL `jobs` table)
      </h3>
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-gray-50">
            <th className="px-6 py-4 text-left text-gray-500 font-medium text-sm uppercase tracking-wider">Job ID</th>
            <th className="px-6 py-4 text-left text-gray-500 font-medium text-sm uppercase tracking-wider">Email User</th>
            <th className="px-6 py-4 text-left text-gray-500 font-medium text-sm uppercase tracking-wider">Tên File / Nguồn</th>
            <th className="px-6 py-4 text-left text-gray-500 font-medium text-sm uppercase tracking-wider">Độ dài</th>
            <th className="px-6 py-4 text-left text-gray-500 font-medium text-sm uppercase tracking-wider">Trạng thái</th>
            <th className="px-6 py-4 text-left text-gray-500 font-medium text-sm uppercase tracking-wider">AI Pipeline</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job, i) => (
            <tr key={i} className="border-b border-gray-200 hover:bg-gray-50 transition-colors">
              <td className="px-6 py-4 font-mono text-gray-500 text-sm">{job.id}</td>
              <td className="px-6 py-4 text-sm">{job.email}</td>
              <td className="px-6 py-4 text-sm">{job.name}</td>
              <td className="px-6 py-4 text-sm">{job.duration}</td>
              <td className="px-6 py-4">
                <span className={`px-2.5 py-1 rounded-md text-xs font-semibold ${statusClasses[job.status]}`}>
                  {job.statusText}
                </span>
              </td>
              <td className="px-6 py-4 text-sm">{job.pipeline}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
