import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { LandingPage } from './pages/LandingPage';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { QaPage } from './pages/QaPage';
import { AdminPage } from './pages/AdminPage';
import { HistoryPage } from './pages/HistoryPage';
import { AuthPage } from './pages/AuthPage';
import { ProfilePage } from './pages/ProfilePage';
import { DocsPage } from './pages/DocsPage';
import { api } from './services/api';
import { ToastProvider } from './context/ToastContext';
import { TopProgress } from './components/TopProgress';
import './App.css';

interface UserState {
  email: string;
  role: string;
}

function App() {
  const [user, setUser] = useState<UserState | null>(() => {
    const stored = localStorage.getItem('user');
    return stored ? JSON.parse(stored) : null;
  });
  const [isLoggedIn, setIsLoggedIn] = useState(() => !!localStorage.getItem('user'));

  // On mount, check if token exists and verify profile with backend if server is up
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      api.getMe()
      .then(res => {
        if (res.success && res.data) {
          const userData = {
            email: res.data.email,
            role: res.data.role.toLowerCase()
          };
          setUser(userData);
          setIsLoggedIn(true);
          localStorage.setItem('user', JSON.stringify(userData));
        }
      })
      .catch(() => {
        // Fallback or offline support
      });
    }
  }, []);

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  const handleLogin = (userData: UserState) => {
    setIsLoggedIn(true);
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  return (
    <ToastProvider>
      <TopProgress />
      <Router>
        <div className="app-container">
          <Header isLoggedIn={isLoggedIn} onLogout={handleLogout} />
          <main className="app-main-content">
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route 
                path="/upload" 
                element={isLoggedIn ? <UploadPage /> : <Navigate to="/auth" />} 
              />
              <Route 
                path="/results" 
                element={isLoggedIn ? <ResultsPage /> : <Navigate to="/auth" />} 
              />
              <Route 
                path="/qa" 
                element={isLoggedIn ? <QaPage /> : <Navigate to="/auth" />} 
              />
              <Route 
                path="/admin/*" 
                element={isLoggedIn ? (user?.role === 'admin' ? <AdminPage /> : <Navigate to="/history" />) : <Navigate to="/auth" />} 
              />
              <Route 
                path="/history" 
                element={isLoggedIn ? <HistoryPage /> : <Navigate to="/auth" />} 
              />
              <Route 
                path="/profile" 
                element={isLoggedIn ? <ProfilePage /> : <Navigate to="/auth" />} 
              />
              <Route 
                path="/auth" 
                element={<AuthPage onLogin={handleLogin} />} 
              />
              <Route path="/docs" element={<DocsPage />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ToastProvider>
  );
}

export default App;
