import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

export const LandingPage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let animationFrameId: number;

    const gl = (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;
    if (!gl) return;

    // Sync WebGL buffer size with client size
    const resizeCanvas = () => {
      const w = canvas.clientWidth || 1280;
      const h = canvas.clientHeight || 720;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const vsSource = `
      attribute vec2 a_position;
      varying vec2 v_texCoord;
      void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    const fsSource = `
      precision highp float;
      uniform float u_time;
      uniform vec2 u_resolution;

      void main() {
          vec2 uv = gl_FragCoord.xy / u_resolution.xy;
          float t = u_time * 0.15;
          
          // Light-themed academic palette
          vec3 color1 = vec3(0.95, 0.96, 0.98); // Very light slate/surface
          vec3 color2 = vec3(0.92, 0.94, 0.96); // Soft gray-blue
          vec3 accent = vec3(0.02, 0.71, 0.83); // Vibrant Cyan (normalized)
          
          // Smooth flowing motion
          float waves = sin(uv.x * 8.0 + t) * cos(uv.y * 6.0 - t * 0.5) * 0.5 + 0.5;
          vec3 baseColor = mix(color1, color2, waves);
          
          // Subtle digital grid (lighter for light mode)
          vec2 grid = fract(uv * 35.0);
          float line = smoothstep(0.0, 0.03, grid.x) * smoothstep(0.0, 0.03, grid.y);
          baseColor = mix(baseColor, baseColor - 0.02, 1.0 - line);
          
          // Very faint light pulses
          float pulse = sin(uv.x * 3.0 - t * 2.0) * cos(uv.y * 2.0 + t) * 0.5 + 0.5;
          baseColor = mix(baseColor, mix(baseColor, accent, 0.03), pulse * 0.2);

          gl_FragColor = vec4(baseColor, 1.0);
      }
    `;

    const compileShader = (source: string, type: number) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vs = compileShader(vsSource, gl.VERTEX_SHADER);
    const fs = compileShader(fsSource, gl.FRAGMENT_SHADER);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program linking error:', gl.getProgramInfoLog(program));
      return;
    }

    gl.useProgram(program);

    // Quad buffer
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    const uTimeLoc = gl.getUniformLocation(program, 'u_time');
    const uResolutionLoc = gl.getUniformLocation(program, 'u_resolution');

    const render = (time: number) => {
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(uTimeLoc, time * 0.001);
      gl.uniform2f(uResolutionLoc, canvas.width, canvas.height);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteBuffer(buffer);
    };
  }, []);

  return (
    <div className="bg-surface text-on-surface">
      {/* Hero Section */}
      <section className="relative h-[85vh] flex items-center justify-center overflow-hidden bg-surface">
        <div className="absolute inset-0 w-full h-full z-0 opacity-80">
          <canvas ref={canvasRef} className="block w-full h-full" />
        </div>
        <div className="relative z-10 text-center max-w-4xl px-6">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary-fixed text-on-primary-fixed font-label-md text-label-md mb-6 animate-pulse">
            <span className="material-symbols-outlined text-[18px]">auto_awesome</span>
            Powered by Multimodal LLMs & Speech Models
          </div>
          <h1 className="font-headline-xl text-4xl md:text-[64px] md:leading-[1.1] mb-6 text-deep-navy tracking-tight font-extrabold">
            Automatic Video <br /> <span className="text-vibrant-cyan">Lecture Summarization</span>
          </h1>
          <p className="font-body-lg text-lg text-slate-600 mb-10 max-w-2xl mx-auto leading-relaxed">
            Tiết kiệm hàng giờ học tập và nghiên cứu. Hệ thống AI đa phương thức tự động đồng bộ hóa hình ảnh, giọng nói, text slide để trích xuất tóm tắt thông tin chất lượng cao từ các bài giảng học thuật.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link 
              to="/upload" 
              className="px-8 py-4 bg-deep-navy text-white font-bold rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-all active:scale-95 animate-fade-in"
            >
              Phân tích Video Ngay
              <span className="material-symbols-outlined">arrow_forward</span>
            </Link>
            <Link 
              to="/results" 
              className="px-8 py-4 border border-outline text-deep-navy font-bold rounded-lg flex items-center justify-center gap-2 hover:bg-surface-container-low transition-all bg-surface/50 backdrop-blur-sm"
            >
              Xem Demo Kết Quả
            </Link>
          </div>
        </div>
        {/* Scroll Indicator */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-50 text-deep-navy">
          <span className="font-label-sm text-xs">Cuộn để khám phá thêm</span>
          <span className="material-symbols-outlined animate-bounce">expand_more</span>
        </div>
      </section>

      {/* About Section: Modular Architecture */}
      <section className="py-24 bg-surface-container-lowest">
        <div className="max-w-container-max mx-auto px-6 md:px-margin-desktop">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <span className="font-label-md text-label-md text-vibrant-cyan tracking-widest uppercase mb-4 block font-semibold">
                Our Core Engine
              </span>
              <h2 className="font-headline-xl text-3xl md:text-headline-xl font-bold text-deep-navy mb-6">
                Kiến trúc đa phương thức (Multimodal Fusion)
              </h2>
              <p className="font-body-md text-body-md text-secondary mb-8 leading-relaxed">
                Hệ thống kết hợp đồng bộ 3 luồng dữ liệu chính để phân tích toàn diện nội dung video, đảm bảo không bỏ sót thông tin cốt lõi quan trọng nào của buổi học.
              </p>
              <div className="space-y-6">
                <div className="flex gap-4 p-4 rounded-xl border border-outline-variant bg-surface hover:border-vibrant-cyan transition-colors">
                  <div className="w-12 h-12 rounded-lg bg-primary-fixed flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-on-primary-fixed">mic</span>
                  </div>
                  <div>
                    <h4 className="font-headline-md text-lg font-bold text-deep-navy mb-1">Xử lý âm thanh</h4>
                    <p className="font-body-sm text-sm text-secondary">
                      Nhận dạng giọng nói (ASR) với độ chính xác cao bằng WhisperX kèm speaker diarization.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4 p-4 rounded-xl border border-outline-variant bg-surface hover:border-vibrant-cyan transition-colors">
                  <div className="w-12 h-12 rounded-lg bg-secondary-fixed flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-on-secondary-fixed">visibility</span>
                  </div>
                  <div>
                    <h4 className="font-headline-md text-lg font-bold text-deep-navy mb-1">Nhận diện hình ảnh</h4>
                    <p className="font-body-sm text-sm text-secondary">
                      Phát hiện cảnh chuyển slide (PySceneDetect), quét nội dung văn bản trên màn hình (PaddleOCR).
                    </p>
                  </div>
                </div>
                <div className="flex gap-4 p-4 rounded-xl border border-outline-variant bg-surface hover:border-vibrant-cyan transition-colors">
                  <div className="w-12 h-12 rounded-lg bg-tertiary-fixed flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-on-tertiary-fixed">hub</span>
                  </div>
                  <div>
                    <h4 className="font-headline-md text-lg font-bold text-deep-navy mb-1">Đồng bộ Fusion RAG</h4>
                    <p className="font-body-sm text-sm text-secondary">
                      Cân chỉnh dữ liệu audio-visual và tạo lập cơ sở tri thức vector trong ChromaDB để tóm tắt và hỗ trợ Q&A.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="relative group">
              <div className="absolute -inset-4 bg-vibrant-cyan/10 blur-2xl rounded-full group-hover:bg-vibrant-cyan/20 transition-all duration-700"></div>
              <div className="relative rounded-2xl overflow-hidden border border-outline-variant shadow-lg bg-deep-navy aspect-video flex items-center justify-center">
                <img 
                  className="w-full h-full object-cover opacity-60" 
                  alt="Neural Network Architecture Visualization" 
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuCjlP2oIEsOV4pXgBsi6yU5Ybtu-XkeF4hjp5e40WZUWvmNjf_yOjnrQCTc9NDDV2Alb7dFewV8aeyda-3jIca56RdBR77xFMuhjKjcIqZnrtcq9ZEoDfkj0_0LyYkYEBRLwFtmN6CRY-r3gPJ8Z9ahLj-MwhsNNIg5XsddCn9P5Wyp3HAmUXqN0Re_An4ckLhhTHosQgurlnFAphJu7huY1JQWG5oDHeqtYq1VKt9kHRhV-Znp45ol"
                />
                <Link to="/results" className="absolute inset-0 flex items-center justify-center group/play">
                  <div className="w-20 h-20 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/20 group-hover/play:scale-110 transition-transform">
                    <span className="material-symbols-outlined text-white text-4xl" style={{ fontVariationSettings: "'FILL' 1" }}>play_arrow</span>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Bento Grid */}
      <section className="py-24 bg-surface-container-low">
        <div className="max-w-container-max mx-auto px-6 md:px-margin-desktop">
          <div className="text-center mb-16">
            <h2 className="font-headline-xl text-3xl md:text-headline-xl font-bold text-deep-navy mb-4">
              Tính năng cấp Nghiên cứu (Research-Grade)
            </h2>
            <p className="font-body-md text-body-md text-secondary">
              Được thiết kế nhằm đáp ứng tốt nhất các yêu cầu học tập, phân tích và trích xuất tri thức bài giảng.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Feature 1: Abstractive Summarization */}
            <div className="md:col-span-2 bg-surface border border-outline-variant p-8 rounded-xl flex flex-col justify-between group hover:border-vibrant-cyan transition-all">
              <div className="max-w-md">
                <div className="w-10 h-10 rounded-lg bg-deep-navy text-white flex items-center justify-center mb-6">
                  <span className="material-symbols-outlined">summarize</span>
                </div>
                <h3 className="font-headline-lg text-xl font-bold text-deep-navy mb-3">Tóm tắt Tổng hợp (Abstractive)</h3>
                <p className="font-body-md text-body-md text-secondary leading-relaxed mb-8">
                  Không chỉ là trích xuất câu từ thô sơ, AI tự tạo lập các đoạn tóm tắt chất lượng, giải thích thuật ngữ chuyên ngành và tổ chức thành mục tiêu bài học mạch lạc.
                </p>
              </div>
              <div className="bg-surface-container-high rounded-lg p-4 border border-outline-variant shadow-sm flex items-center gap-4">
                <div className="flex -space-x-2">
                  <div className="w-8 h-8 rounded-full border-2 border-white bg-slate-200"></div>
                  <div className="w-8 h-8 rounded-full border-2 border-white bg-slate-300"></div>
                  <div className="w-8 h-8 rounded-full border-2 border-white bg-slate-400"></div>
                </div>
                <span className="font-label-sm text-sm text-secondary font-medium">Hỗ trợ các viện nghiên cứu và đại học lớn</span>
              </div>
            </div>

            {/* Feature 2: Smart Chaptering */}
            <div className="bg-primary-container p-8 rounded-xl border border-slate-600 flex flex-col justify-between group hover:shadow-xl transition-all">
              <div>
                <div className="w-10 h-10 rounded-lg bg-vibrant-cyan text-white flex items-center justify-center mb-6">
                  <span className="material-symbols-outlined">view_day</span>
                </div>
                <h3 className="font-headline-lg text-xl font-bold text-white mb-3">Phân chia chương học</h3>
                <p className="font-body-md text-body-md text-slate-300 leading-relaxed">
                  Tự động cắt nhỏ video thành các chương nội dung (chapter) tương ứng với các slide và chủ đề thảo luận chính trong bài giảng.
                </p>
              </div>
              <div className="mt-8 space-y-2">
                <div className="h-1 bg-slate-700 w-full rounded-full overflow-hidden">
                  <div className="h-full bg-vibrant-cyan w-1/3"></div>
                </div>
                <div className="flex justify-between font-mono-data text-[10px] text-vibrant-cyan">
                  <span>04:12</span>
                  <span>12:45</span>
                  <span>21:02</span>
                </div>
              </div>
            </div>

            {/* Feature 3: Q&A Chat */}
            <div className="bg-surface border border-outline-variant p-8 rounded-xl flex flex-col group hover:border-vibrant-cyan transition-all">
              <div className="w-10 h-10 rounded-lg bg-secondary-container text-primary flex items-center justify-center mb-6">
                <span className="material-symbols-outlined text-deep-navy">chat_bubble</span>
              </div>
              <h3 className="font-headline-lg text-xl font-bold text-deep-navy mb-3">Hỏi đáp Bài giảng RAG</h3>
              <p className="font-body-md text-body-md text-secondary leading-relaxed mb-6">
                Trực tiếp chat hỏi đáp thông tin. Trích dẫn câu trả lời dựa trên ngữ cảnh video và slide với timestamp liên kết.
              </p>
              <div className="mt-auto p-3 bg-slate-50 border border-outline-variant rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-status-success"></div>
                  <span className="font-label-sm text-xs text-deep-navy font-bold">RAG Engine Active</span>
                </div>
                <p className="font-body-sm text-xs italic text-secondary">"Giải thích cấu trúc thuật toán ở phút 15:20..."</p>
              </div>
            </div>

            {/* Feature 4: Metric Insights */}
            <div className="md:col-span-2 bg-deep-navy p-8 rounded-xl border border-slate-600 flex flex-col md:flex-row gap-8 items-center text-white">
              <div className="md:w-1/2">
                <h3 className="font-headline-lg text-xl font-bold mb-3">Phân tích thời gian thực</h3>
                <p className="font-body-md text-body-md text-slate-300 leading-relaxed">
                  Theo dõi trạng thái GPU cluster, tốc độ xử lý âm thanh hình ảnh, tỷ lệ nhận diện chữ viết (OCR) trực quan.
                </p>
              </div>
              <div className="md:w-1/2 grid grid-cols-2 gap-4 w-full">
                <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
                  <div className="text-vibrant-cyan text-headline-lg font-bold">98.4%</div>
                  <div className="font-label-sm text-xs text-slate-400">Độ chính xác OCR</div>
                </div>
                <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
                  <div className="text-status-success text-headline-lg font-bold">4x</div>
                  <div className="font-label-sm text-xs text-slate-400">Tốc độ xử lý (GPU)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6 md:px-margin-desktop overflow-hidden bg-surface-container-lowest">
        <div className="max-w-container-max mx-auto relative rounded-3xl bg-primary-container p-12 md:p-20 text-center overflow-hidden border border-slate-700">
          <div className="absolute top-0 right-0 w-1/2 h-full opacity-10 pointer-events-none">
            <svg height="100%" preserveAspectRatio="none" viewBox="0 0 100 100" width="100%">
              <defs>
                <pattern height="10" id="grid" patternUnits="userSpaceOnUse" width="10">
                  <path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" strokeWidth="0.5"></path>
                </pattern>
              </defs>
              <rect fill="url(#grid)" height="100%" width="100%"></rect>
            </svg>
          </div>
          <div className="relative z-10">
            <h2 className="font-headline-xl text-3xl md:text-headline-xl font-bold text-white mb-6">
              Sẵn sàng tối ưu hóa việc học tập của bạn?
            </h2>
            <p className="font-body-lg text-lg text-slate-300 max-w-2xl mx-auto mb-10">
              Trải nghiệm hệ thống ngay hôm nay để nâng cao hiệu suất đọc hiểu bài giảng trực tuyến và hội thảo khoa học của bạn.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link 
                to="/upload" 
                className="px-10 py-5 bg-vibrant-cyan text-deep-navy font-bold rounded-lg hover:scale-105 transition-transform"
              >
                Khởi tạo tài khoản Miễn phí
              </Link>
              <Link 
                to="/docs" 
                className="px-10 py-5 border border-slate-500 text-white rounded-lg hover:bg-slate-800 transition-colors"
              >
                Tài liệu Kỹ thuật RAG
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-surface-container-lowest text-secondary font-body-sm text-sm border-t border-outline-variant">
        <div className="flex flex-col md:flex-row justify-between items-center px-6 md:px-margin-desktop py-8 w-full max-w-container-max mx-auto">
          <div className="flex flex-col gap-2 mb-6 md:mb-0">
            <div className="font-label-md text-label-md font-bold text-deep-navy">Multimodal Lecture Summarizer</div>
            <p className="max-w-xs text-xs text-secondary">
              © 2026 Institute for Multimodal AI Research. All rights reserved.
            </p>
          </div>
          <div className="flex flex-wrap gap-6 justify-center">
            <Link className="text-secondary hover:text-primary transition-colors text-xs" to="/docs">Academic Credits</Link>
            <Link className="text-secondary hover:text-primary transition-colors text-xs" to="#">Privacy Policy</Link>
            <Link className="text-secondary hover:text-primary transition-colors text-xs" to="#">Terms of Service</Link>
          </div>
          <div className="mt-6 md:mt-0 flex gap-4">
            <div className="w-8 h-8 rounded-full border border-outline-variant flex items-center justify-center hover:bg-surface-container-high transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-sm">share</span>
            </div>
            <div className="w-8 h-8 rounded-full border border-outline-variant flex items-center justify-center hover:bg-surface-container-high transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-sm">public</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
