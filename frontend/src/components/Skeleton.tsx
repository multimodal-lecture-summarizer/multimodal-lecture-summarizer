import React from 'react';

interface SkeletonProps {
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> & {
  Text: React.FC<{ lines?: number; className?: string }>;
  Card: React.FC<SkeletonProps>;
  TableRow: React.FC<{ cells?: number; className?: string }>;
  ChatBubble: React.FC<{ isBot?: boolean; className?: string }>;
} = ({ className = '' }) => {
  return (
    <div className={`animate-pulse bg-slate-300/80 dark:bg-slate-600 rounded ${className}`} />
  );
};

// Text Skeleton
Skeleton.Text = ({ lines = 3, className = '' }) => {
  return (
    <div className={`space-y-2.5 ${className}`}>
      {Array.from({ length: lines }).map((_, idx) => {
        // Vary the width of text lines for a more realistic paragraph look
        let widthClass = 'w-full';
        if (idx === lines - 1 && lines > 1) {
          widthClass = 'w-3/5';
        } else if (idx === 0) {
          widthClass = 'w-11/12';
        }
        return (
          <Skeleton
            key={idx}
            className={`h-4 ${widthClass}`}
          />
        );
      })}
    </div>
  );
};

// Card Skeleton
Skeleton.Card = ({ className = '' }) => {
  return (
    <div className={`bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm flex flex-col p-4 space-y-4 ${className}`}>
      <Skeleton className="h-44 w-full rounded-xl" />
      <div className="space-y-3 flex-1 flex flex-col justify-between">
        <div>
          <Skeleton className="h-5 w-3/4 rounded-md mb-2" />
          <Skeleton className="h-4 w-1/2 rounded-md" />
        </div>
        <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
          <div className="flex gap-2">
            <Skeleton className="w-8 h-8 rounded-lg" />
            <Skeleton className="w-8 h-8 rounded-lg" />
          </div>
          <Skeleton className="w-8 h-8 rounded-lg" />
        </div>
      </div>
    </div>
  );
};

// TableRow Skeleton
Skeleton.TableRow = ({ cells = 5, className = '' }) => {
  return (
    <tr className={`border-b border-slate-100 ${className}`}>
      {Array.from({ length: cells }).map((_, idx) => (
        <td key={idx} className="p-4">
          <Skeleton className={`h-4 ${idx === 0 ? 'w-24' : 'w-16'}`} />
        </td>
      ))}
    </tr>
  );
};

// ChatBubble Skeleton
Skeleton.ChatBubble = ({ isBot = true, className = '' }) => {
  return (
    <div className={`flex gap-3 items-start ${isBot ? 'flex-row' : 'flex-row-reverse'} ${className}`}>
      <Skeleton className="w-8 h-8 rounded-full shrink-0" />
      <div className={`max-w-[70%] p-4 rounded-2xl space-y-2 ${
        isBot 
          ? 'bg-slate-100 rounded-tl-none' 
          : 'bg-indigo-50 rounded-tr-none'
      }`}>
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-3 w-28" />
      </div>
    </div>
  );
};
