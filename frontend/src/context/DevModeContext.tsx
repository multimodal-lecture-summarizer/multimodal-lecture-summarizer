import React, { createContext, useContext, useState } from 'react';

interface DevModeContextType {
  isDevMode: boolean;
  toggleDevMode: () => void;
  setDevMode: (val: boolean) => void;
}

const DevModeContext = createContext<DevModeContextType>({
  isDevMode: false,
  toggleDevMode: () => {},
  setDevMode: () => {},
});

export const DevModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDevMode, setIsDevMode] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem('prism_dev_mode');
      return stored === 'true';
    } catch {
      return false;
    }
  });

  const toggleDevMode = () => {
    setIsDevMode((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('prism_dev_mode', String(next));
      } catch (e) {
        console.warn('Failed to save devMode preference', e);
      }
      return next;
    });
  };

  const setDevMode = (val: boolean) => {
    setIsDevMode(val);
    try {
      localStorage.setItem('prism_dev_mode', String(val));
    } catch (e) {
      console.warn('Failed to save devMode preference', e);
    }
  };

  return (
    <DevModeContext.Provider value={{ isDevMode, toggleDevMode, setDevMode }}>
      {children}
    </DevModeContext.Provider>
  );
};

export const useDevMode = () => useContext(DevModeContext);
