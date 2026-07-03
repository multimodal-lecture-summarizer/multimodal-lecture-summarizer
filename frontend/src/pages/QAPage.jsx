/**
 * QAPage — Trang hỏi đáp RAG với video.
 * Trang: UIs/4_qa.html → frontend/src/pages/QAPage.jsx
 */
import VideoSidebar from '../components/VideoSidebar';
import ChatInterface from '../components/ChatInterface';

export default function QAPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <VideoSidebar />
      <ChatInterface />
    </div>
  );
}
