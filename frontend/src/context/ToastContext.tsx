import React, { createContext, useContext, useState, useCallback } from 'react';

export type ToastType = 'success' | 'error' | 'info';

export interface ToastMessage {
  id: string;
  message: string;
  title?: string;
  type: ToastType;
  duration?: number;
}

interface ToastContextProps {
  showToast: (message: string, type?: ToastType, title?: string, duration?: number) => void;
  success: (message: string, title?: string, duration?: number) => void;
  error: (message: string, title?: string, duration?: number) => void;
  info: (message: string, title?: string, duration?: number) => void;
}

const ToastContext = createContext<ToastContextProps | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    (message: string, type: ToastType = 'info', title?: string, duration = 4000) => {
      const id = Math.random().toString(36).substring(2, 9);
      
      let defaultTitle = '';
      if (!title) {
        if (type === 'success') defaultTitle = 'Success';
        else if (type === 'error') defaultTitle = 'Error';
        else defaultTitle = 'Notification';
      } else {
        defaultTitle = title;
      }

      setToasts((prev) => [...prev, { id, message, type, title: defaultTitle, duration }]);

      if (duration > 0) {
        setTimeout(() => {
          dismissToast(id);
        }, duration);
      }
    },
    [dismissToast]
  );

  const success = useCallback((msg: string, title?: string, dur?: number) => showToast(msg, 'success', title, dur), [showToast]);
  const error = useCallback((msg: string, title?: string, dur?: number) => showToast(msg, 'error', title, dur), [showToast]);
  const info = useCallback((msg: string, title?: string, dur?: number) => showToast(msg, 'info', title, dur), [showToast]);

  return (
    <ToastContext.Provider value={{ showToast, success, error, info }}>
      {children}
      
      {/* Toast Overlay Container */}
      <div className="fixed bottom-8 right-8 flex flex-col gap-4 z-[100] w-full max-w-sm pointer-events-none">
        {toasts.map((toast) => {
          let leftBorderClass = 'bg-vibrant-cyan';
          let iconSvg = (
            <svg className="w-6 h-6 text-vibrant-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          );

          if (toast.type === 'success') {
            leftBorderClass = 'bg-status-success';
            iconSvg = (
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM10 17L5 12L6.41 10.59L10 14.17L17.59 6.58L19 8L10 17Z" fill="#10B981"/>
              </svg>
            );
          } else if (toast.type === 'error') {
            leftBorderClass = 'bg-error';
            iconSvg = (
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="#EF4444"/>
              </svg>
            );
          }

          return (
            <div
              key={toast.id}
              className="toast-entrance bg-white border border-outline-variant rounded-lg shadow-lg overflow-hidden flex items-stretch pointer-events-auto transition-all duration-300"
            >
              <div className={`${leftBorderClass} w-1.5 shrink-0`}></div>
              <div className="flex-1 flex items-start p-4 gap-3">
                <div className="shrink-0 mt-0.5">{iconSvg}</div>
                <div className="flex-1">
                  <div className="font-label-md text-label-md text-deep-navy font-bold">{toast.title}</div>
                  <div className="font-body-sm text-body-sm text-secondary mt-0.5">{toast.message}</div>
                </div>
                <button
                  onClick={() => dismissToast(toast.id)}
                  className="text-secondary hover:text-primary transition-colors mt-0.5 shrink-0"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};
