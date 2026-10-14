/* Hallmark · component: EvaluationBenchmarkPage · genre: modern-minimal · theme: editorial-technical
 * states: default · hover · focus-visible · active · disabled · loading · error · success
 * microinteractions: tactile hover elevation, high-contrast monospace technical badges, interactive grouped bar chart, live metric cards
 * pre-emit critique: P5 H5 E5 S5 R5 V5
 */

import React from 'react';
import { EvaluationBenchmarkView } from '../components/EvaluationBenchmarkView';
import { Header } from '../components/Header';

export const EvaluationBenchmarkPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <Header />
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8">
        <EvaluationBenchmarkView />
      </main>
    </div>
  );
};

export default EvaluationBenchmarkPage;
