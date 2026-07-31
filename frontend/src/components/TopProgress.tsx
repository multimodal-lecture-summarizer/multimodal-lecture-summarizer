import React, { useState, useEffect } from 'react';

export const TopProgress: React.FC = () => {
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    let requests = 0;
    let timer: any;
    
    const handleStart = () => {
      requests++;
      if (requests === 1) {
        setVisible(true);
        setProgress(15);
        // fake progress
        timer = setInterval(() => {
          setProgress(p => {
            const inc = (100 - p) * 0.15;
            return p + inc;
          });
        }, 200);
      }
    };
    
    const handleEnd = () => {
      requests = Math.max(0, requests - 1);
      if (requests === 0) {
        setProgress(100);
        clearInterval(timer);
        setTimeout(() => {
          setVisible(false);
          setTimeout(() => setProgress(0), 300);
        }, 400);
      }
    };

    window.addEventListener('api-request-start', handleStart);
    window.addEventListener('api-request-end', handleEnd);

    return () => {
      window.removeEventListener('api-request-start', handleStart);
      window.removeEventListener('api-request-end', handleEnd);
      clearInterval(timer);
    };
  }, []);

  if (!visible && progress === 0) return null;

  return (
    <div className="fixed top-0 left-0 w-full h-[3px] z-[9999] pointer-events-none">
      <div 
        className="h-full bg-primary transition-all duration-300 ease-out"
        style={{ 
          width: `${progress}%`, 
          opacity: visible ? 1 : 0,
          boxShadow: '0 0 10px var(--primary), 0 0 5px var(--primary)'
        }}
      />
    </div>
  );
};
