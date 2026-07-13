import React from 'react';
import { motion } from 'framer-motion';

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
    <motion.div 
      initial={{ opacity: 0.5 }}
      animate={{ opacity: [0.5, 0.8, 0.5] }}
      transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
      className={`bg-slate-200/50 rounded-lg backdrop-blur-sm ${className}`} 
    />
  );
};

// Text Skeleton
Skeleton.Text = ({ lines = 3, className = '' }) => {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: lines }).map((_, idx) => {
        let widthClass = 'w-full';
        if (idx === lines - 1 && lines > 1) widthClass = 'w-3/5';
        else if (idx === 0) widthClass = 'w-11/12';
        return <Skeleton key={idx} className={`h-3 ${widthClass}`} />;
      })}
    </div>
  );
};

// Card Skeleton
Skeleton.Card = ({ className = '' }) => {
  return (
    <div className={`glass p-6 rounded-2xl flex flex-col space-y-4 ${className}`}>
      <Skeleton className="h-48 w-full rounded-xl" />
      <div className="space-y-4 flex-1 flex flex-col justify-between">
        <div>
          <Skeleton className="h-6 w-3/4 mb-3" />
          <Skeleton className="h-4 w-1/2" />
        </div>
        <div className="pt-4 flex items-center justify-between">
          <div className="flex gap-2">
            <Skeleton className="w-8 h-8 rounded-full" />
            <Skeleton className="w-8 h-8 rounded-full" />
          </div>
          <Skeleton className="w-20 h-8 rounded-lg" />
        </div>
      </div>
    </div>
  );
};

// TableRow Skeleton
Skeleton.TableRow = ({ cells = 5, className = '' }) => {
  return (
    <tr className={`border-b border-slate-100/50 ${className}`}>
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
    <div className={`flex gap-3 items-end ${isBot ? 'flex-row' : 'flex-row-reverse'} ${className}`}>
      <Skeleton className="w-8 h-8 rounded-full shrink-0" />
      <div className={`max-w-[70%] p-4 rounded-2xl space-y-3 ${
        isBot ? 'glass rounded-bl-none' : 'bg-primary/10 rounded-br-none'
      }`}>
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-4 w-32" />
      </div>
    </div>
  );
};
