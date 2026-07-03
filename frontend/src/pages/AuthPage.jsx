/**
 * AuthPage — Trang đăng nhập.
 * Trang: UIs/7_auth.html → frontend/src/pages/AuthPage.jsx
 */
import AuthForm from '../components/AuthForm';

export default function AuthPage() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <AuthForm />
    </div>
  );
}
