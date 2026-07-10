import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

export const DocsPage: React.FC = () => {
  const { t } = useTranslation();
  const [activeSection, setActiveSection] = useState<'overview' | 'summary' | 'keyframe' | 'rag'>('overview');

  return (
    <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-slate-50 text-slate-900">
      {/* Sidebar for documentation navigation */}
      <aside className="w-64 border-r-2 border-slate-200 bg-white flex flex-col shrink-0 p-4 gap-2">
        <div className="px-2 py-4 border-b-2 border-slate-100 mb-2 flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">menu_book</span>
          <span className="font-heading text-sm font-bold text-slate-900">{t('docs.title')}</span>
        </div>
        <nav className="flex flex-col gap-1 flex-1">
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left text-xs font-bold ${
              activeSection === 'overview' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
            }`}
            onClick={() => setActiveSection('overview')}
          >
            <span className="material-symbols-outlined text-sm">home_storage</span>
            {t('docs.overview')}
          </button>
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left text-xs font-bold ${
              activeSection === 'summary' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
            }`}
            onClick={() => setActiveSection('summary')}
          >
            <span className="material-symbols-outlined text-sm">article</span>
            {t('docs.summary')}
          </button>
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left text-xs font-bold ${
              activeSection === 'keyframe' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
            }`}
            onClick={() => setActiveSection('keyframe')}
          >
            <span className="material-symbols-outlined text-sm">image</span>
            {t('docs.keyframe')}
          </button>
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left text-xs font-bold ${
              activeSection === 'rag' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
            }`}
            onClick={() => setActiveSection('rag')}
          >
            <span className="material-symbols-outlined text-sm">forum</span>
            {t('docs.rag')}
          </button>
        </nav>
      </aside>

      {/* Main content display area */}
      <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-50 custom-scrollbar">
        <div className="max-w-3xl mx-auto bg-white border-2 border-slate-200 rounded-xl p-8 shadow-none">
          
          {/* SECTION 1: OVERVIEW */}
          {activeSection === 'overview' && (
            <div className="space-y-6">
              <h1 className="font-heading text-2xl font-bold text-slate-900">{t('docs.overview_title')}</h1>
              <p className="text-slate-500 text-sm leading-relaxed">
                {t('docs.overview_desc1')} <strong className="text-slate-900">{t('docs.overview_desc2')}</strong>{t('docs.overview_desc3')}
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-5 border-2 border-slate-200 rounded-xl bg-slate-50 space-y-2">
                  <span className="material-symbols-outlined text-primary text-3xl">magic_button</span>
                  <h3 className="text-xs font-bold text-slate-900">{t('docs.feat1_title')}</h3>
                  <p className="text-[11px] text-slate-500 leading-normal">{t('docs.feat1_desc')}</p>
                </div>
                <div className="p-5 border-2 border-slate-200 rounded-xl bg-slate-50 space-y-2">
                  <span className="material-symbols-outlined text-primary text-3xl">history_toggle_off</span>
                  <h3 className="text-xs font-bold text-slate-900">{t('docs.feat2_title')}</h3>
                  <p className="text-[11px] text-slate-500 leading-normal">{t('docs.feat2_desc')}</p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-slate-900 pt-2">{t('docs.ai_process_title')}</h2>
              <div className="space-y-4">
                <div className="flex gap-4 items-start">
                  <div className="w-6 h-6 rounded-full bg-slate-900 text-primary flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">1</div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900">{t('docs.step1_title')}</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">{t('docs.step1_desc')}</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-6 h-6 rounded-full bg-slate-900 text-primary flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">2</div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900">{t('docs.step2_title')}</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">{t('docs.step2_desc')}</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-6 h-6 rounded-full bg-slate-900 text-primary flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">3</div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900">{t('docs.step3_title')}</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">{t('docs.step3_desc')}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SECTION 2: TRANSCRIPTION & SUMMARY */}
          {activeSection === 'summary' && (
            <div className="space-y-6">
              <h1 className="font-heading text-2xl font-bold text-slate-900">{t('docs.summary_title')}</h1>
              <p className="text-slate-500 text-sm leading-relaxed">
                {t('docs.summary_desc')}
              </p>
              
              <div className="flex gap-4 items-start p-4 bg-slate-50 border-2 border-slate-200 rounded-xl">
                <span className="material-symbols-outlined text-primary text-2xl mt-0.5">translate</span>
                <div>
                  <h3 className="text-xs font-bold text-slate-900 mb-1">{t('docs.whisper_title')}</h3>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    {t('docs.whisper_desc')}
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start p-4 bg-slate-50 border-2 border-slate-200 rounded-xl">
                <span className="material-symbols-outlined text-primary text-2xl mt-0.5">view_timeline</span>
                <div>
                  <h3 className="text-xs font-bold text-slate-900 mb-1">{t('docs.chapter_title')}</h3>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    {t('docs.chapter_desc')}
                  </p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-slate-900 pt-2">{t('docs.usage_title')}</h2>
              <ul className="list-disc pl-5 text-xs text-slate-500 space-y-1.5 leading-relaxed">
                <li>{t('docs.usage_step1')}</li>
                <li>{t('docs.usage_step2')}</li>
                <li>{t('docs.usage_step3')}</li>
              </ul>
            </div>
          )}

          {/* SECTION 3: KEYFRAME EXTRACTION */}
          {activeSection === 'keyframe' && (
            <div className="space-y-6">
              <h1 className="font-heading text-2xl font-bold text-slate-900">{t('docs.keyframe_title')}</h1>
              <p className="text-slate-500 text-sm leading-relaxed">
                {t('docs.keyframe_desc')}
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-5 border-2 border-slate-200 rounded-xl bg-slate-50 space-y-2">
                  <h3 className="text-xs font-bold text-slate-900">{t('docs.kf_step1_title')}</h3>
                  <p className="text-[11px] text-slate-500 leading-relaxed">{t('docs.kf_step1_desc')}</p>
                </div>
                <div className="p-5 border-2 border-slate-200 rounded-xl bg-slate-50 space-y-2">
                  <h3 className="text-xs font-bold text-slate-900">{t('docs.kf_step2_title')}</h3>
                  <p className="text-[11px] text-slate-500 leading-relaxed">{t('docs.kf_step2_desc')}</p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-slate-900 pt-2">{t('docs.kf_usage_title')}</h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                {t('docs.kf_usage_desc')}
              </p>
            </div>
          )}

          {/* SECTION 4: INTERACTIVE Q&A */}
          {activeSection === 'rag' && (
            <div className="space-y-6">
              <h1 className="font-heading text-2xl font-bold text-slate-900">{t('docs.rag_title')}</h1>
              <p className="text-slate-500 text-sm leading-relaxed">
                {t('docs.rag_desc')}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="p-4 border-2 border-slate-200 rounded-xl bg-slate-50 text-center space-y-2">
                  <span className="material-symbols-outlined text-primary text-2xl">database</span>
                  <h4 className="text-xs font-bold text-slate-900">{t('docs.rag_box1_title')}</h4>
                  <p className="text-[10px] text-slate-500 leading-relaxed">{t('docs.rag_box1_desc')}</p>
                </div>
                <div className="p-4 border-2 border-slate-200 rounded-xl bg-slate-50 text-center space-y-2">
                  <span className="material-symbols-outlined text-primary text-2xl">search_check</span>
                  <h4 className="text-xs font-bold text-slate-900">{t('docs.rag_box2_title')}</h4>
                  <p className="text-[10px] text-slate-500 leading-relaxed">{t('docs.rag_box2_desc')}</p>
                </div>
                <div className="p-4 border-2 border-slate-200 rounded-xl bg-slate-50 text-center space-y-2">
                  <span className="material-symbols-outlined text-primary text-2xl">smart_toy</span>
                  <h4 className="text-xs font-bold text-slate-900">{t('docs.rag_box3_title')}</h4>
                  <p className="text-[10px] text-slate-500 leading-relaxed">{t('docs.rag_box3_desc')}</p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-slate-900 pt-2">{t('docs.rag_usage_title')}</h2>
              <ol className="list-decimal pl-5 text-xs text-slate-500 space-y-1.5 leading-relaxed">
                <li>{t('docs.rag_usage_step1')}</li>
                <li>{t('docs.rag_usage_step2')}</li>
                <li>{t('docs.rag_usage_step3')}</li>
              </ol>
            </div>
          )}



        </div>
      </div>
    </div>
  );
};
