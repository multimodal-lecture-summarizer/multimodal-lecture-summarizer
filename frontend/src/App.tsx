import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState } from 'react';
import { Header } from './components/Header';
import { LandingPage } from './pages/LandingPage';
import { UploadPage } from './pages/UploadPage';
import { ResultsPage } from './pages/ResultsPage';
import { QaPage } from './pages/QaPage';
import { AdminPage } from './pages/AdminPage';
import { HistoryPage } from './pages/HistoryPage';
import { AuthPage } from './pages/AuthPage';
import './App.css';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(true); // Default to true for easy demo access

  const handleLogout = () => {
    setIsLoggedIn(false);
  };

  const handleLogin = () => {
    setIsLoggedIn(true);
  };

  return (
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
              element={isLoggedIn ? <AdminPage /> : <Navigate to="/auth" />} 
            />
            <Route 
              path="/history" 
              element={isLoggedIn ? <HistoryPage /> : <Navigate to="/auth" />} 
            />
            <Route 
              path="/auth" 
              element={<AuthPage onLogin={handleLogin} />} 
            />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
