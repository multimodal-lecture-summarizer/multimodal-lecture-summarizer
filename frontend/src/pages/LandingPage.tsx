import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { 
  Play, 
  FileText, 
  MessageSquare, 
  Sparkles, 
  ArrowRight,
  Clock,
  Layers,
  Zap
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  const { t } = useTranslation();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#FAF5FF]">
      {/* Decorative Background Elements */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-primary/20 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[10%] right-[-5%] w-[30%] h-[50%] rounded-full bg-blue-400/10 blur-[100px] pointer-events-none" />
      <div className="absolute top-[40%] left-[60%] w-[20%] h-[20%] rounded-full bg-fuchsia-400/20 blur-[80px] pointer-events-none" />

      {/* Hero Section */}
      <motion.section 
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="relative pt-32 pb-24 px-6 md:px-8 text-center max-w-6xl mx-auto z-10"
      >
        <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-primary/20 text-primary font-bold text-sm mb-8 shadow-sm">
          <Sparkles size={16} className="animate-pulse" />
          {t('landing.powered_by') || 'Powered by Advanced AI'}
        </motion.div>
        
        <motion.h1 variants={itemVariants} className="font-heading text-6xl md:text-8xl font-black text-slate-900 mb-6 tracking-tight leading-[1.1]">
          {t('landing.hero_title_1') || 'Understand videos'} <br /> 
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-600">
            {t('landing.hero_title_2') || 'in seconds'}
          </span>
        </motion.h1>
        
        <motion.p variants={itemVariants} className="font-body text-xl md:text-2xl text-slate-600 mb-12 max-w-3xl mx-auto leading-relaxed">
          {t('landing.hero_desc') || 'Paste a YouTube URL or upload a video file to generate a comprehensive summary, key takeaways, and chat with the content.'}
        </motion.p>
        
        <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Link 
            to="/upload" 
            className="group relative inline-flex items-center justify-center gap-3 px-8 py-4 text-lg font-bold text-white bg-primary rounded-xl overflow-hidden transition-all hover:scale-105 active:scale-95 shadow-[0_10px_40px_-10px_rgba(124,58,237,0.5)]"
          >
            <span className="absolute inset-0 w-full h-full -mt-1 rounded-lg opacity-30 bg-gradient-to-b from-transparent via-transparent to-black" />
            <span className="relative flex items-center gap-2">
              {t('landing.analyze_btn') || 'Start Summarizing'}
              <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </span>
          </Link>
          <Link 
            to="/docs" 
            className="group inline-flex items-center justify-center gap-2 px-8 py-4 text-lg font-bold text-slate-700 bg-white/50 backdrop-blur-md rounded-xl border border-slate-200 transition-all hover:bg-white hover:shadow-lg"
          >
            <Play size={20} className="text-primary group-hover:scale-110 transition-transform" />
            Watch Demo
          </Link>
        </motion.div>
      </motion.section>

      {/* Bento Grid Features */}
      <section className="py-24 relative z-10">
        <div className="max-w-6xl mx-auto px-6 md:px-8">
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="font-heading text-4xl md:text-5xl font-black text-slate-900 mb-4">
              {t('landing.features_title') || 'Key Features'}
            </h2>
            <p className="font-body text-xl text-slate-500">
              {t('landing.features_desc') || 'Everything you need to extract knowledge from video content.'}
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[280px]">
            {/* Feature 1 - Large spanning */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              whileHover={{ y: -5 }}
              className="md:col-span-2 glass-panel rounded-3xl p-8 relative overflow-hidden group"
            >
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-[60px] group-hover:bg-primary/20 transition-colors" />
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary to-indigo-600 text-white flex items-center justify-center mb-6 shadow-lg">
                <FileText size={28} />
              </div>
              <h3 className="font-heading text-3xl font-bold text-slate-900 mb-4">{t('landing.abstractive') || 'Abstractive Summarization'}</h3>
              <p className="font-body text-lg text-slate-600 leading-relaxed max-w-md">
                {t('landing.abstractive_desc') || 'Get concise and accurate summaries of long videos. Our AI digests hours of content into minutes of reading.'}
              </p>
            </motion.div>

            {/* Feature 2 - Small */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              whileHover={{ y: -5 }}
              className="glass-panel rounded-3xl p-8 relative overflow-hidden group"
            >
              <div className="w-12 h-12 rounded-xl bg-blue-100 text-blue-600 flex items-center justify-center mb-6">
                <Layers size={24} />
              </div>
              <h3 className="font-heading text-xl font-bold text-slate-900 mb-3">{t('landing.chaptering') || 'Smart Chaptering'}</h3>
              <p className="font-body text-slate-600 leading-relaxed">
                {t('landing.chaptering_desc') || 'Automatically divide videos into logical chapters with timestamps.'}
              </p>
            </motion.div>

            {/* Feature 3 - Small */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              whileHover={{ y: -5 }}
              className="glass-panel rounded-3xl p-8 relative overflow-hidden group"
            >
              <div className="w-12 h-12 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center mb-6">
                <Clock size={24} />
              </div>
              <h3 className="font-heading text-xl font-bold text-slate-900 mb-3">Time Saving</h3>
              <p className="font-body text-slate-600 leading-relaxed">
                Skip the fluff. Jump directly to the moments that matter most in the video.
              </p>
            </motion.div>

            {/* Feature 4 - Large spanning */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              whileHover={{ y: -5 }}
              className="md:col-span-2 glass-panel rounded-3xl p-8 relative overflow-hidden group bg-gradient-to-r from-white/80 to-slate-50/80"
            >
               <div className="absolute bottom-0 right-0 w-64 h-64 bg-fuchsia-400/10 rounded-full blur-[60px] group-hover:bg-fuchsia-400/20 transition-colors" />
              <div className="w-14 h-14 rounded-2xl bg-slate-900 text-white flex items-center justify-center mb-6 shadow-lg">
                <MessageSquare size={28} />
              </div>
              <h3 className="font-heading text-3xl font-bold text-slate-900 mb-4">{t('landing.qa_rag') || 'Q&A Chat (RAG)'}</h3>
              <p className="font-body text-lg text-slate-600 leading-relaxed max-w-md">
                {t('landing.qa_rag_desc') || 'Ask questions and get answers directly from the video content. It is like having a personal tutor for every lecture.'}
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 relative z-10">
        <div className="max-w-5xl mx-auto px-6 md:px-8">
          <motion.div 
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="glass-panel !bg-slate-900/95 rounded-[3rem] p-12 md:p-20 text-center relative overflow-hidden"
          >
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[80%] bg-primary/30 rounded-full blur-[100px] pointer-events-none" />
            <h2 className="font-heading text-4xl md:text-6xl font-black text-white mb-6 relative z-10">
              {t('landing.cta_title') || 'Ready to summarize?'}
            </h2>
            <p className="font-body text-xl text-slate-300 max-w-2xl mx-auto mb-10 relative z-10">
              {t('landing.cta_desc') || 'Join thousands of users who are already saving hours of their time.'}
            </p>
            <div className="flex justify-center relative z-10">
              <Link 
                to="/upload" 
                className="group inline-flex items-center gap-2 bg-white text-slate-900 px-10 py-5 rounded-2xl font-bold text-lg hover:bg-primary-light hover:text-primary transition-colors shadow-xl"
              >
                <Zap size={20} className="text-yellow-500" />
                {t('landing.create_account') || 'Get Started Now'}
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200/50 bg-white/50 backdrop-blur-md relative z-10">
        <div className="max-w-6xl mx-auto px-6 md:px-8 py-10 flex flex-col md:flex-row justify-between items-center text-slate-500 text-sm font-medium">
          <div className="mb-4 md:mb-0 flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-slate-900 flex items-center justify-center">
              <Sparkles size={12} className="text-white" />
            </div>
            <span className="font-heading font-bold text-slate-900 text-lg">PrismVideo</span>
          </div>
          <div className="flex gap-8">
            <Link to="/docs" className="hover:text-primary transition-colors">{t('landing.academic_credits') || 'Credits'}</Link>
            <Link to="#" className="hover:text-primary transition-colors">Privacy</Link>
            <Link to="#" className="hover:text-primary transition-colors">Terms</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};
