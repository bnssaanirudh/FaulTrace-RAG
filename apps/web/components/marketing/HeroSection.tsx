'use client';

import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

export function HeroSection() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.7 } }
  };

  return (
    <section className="relative overflow-hidden bg-theme-bg transition-colors duration-700 pb-20 text-white min-h-[90vh] flex flex-col justify-center pt-24" data-nav-theme="dark">
      {/* Background SVG Grid matching Spur */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.5 }}
        className="pointer-events-none absolute h-screen w-full [&_svg]:object-cover z-0 inset-0"
      >
        <svg className="" width="100%" height="100%" viewBox="0 0 1512 1010" preserveAspectRatio="xMidYMid slice" fill="none" xmlns="http://www.w3.org/2000/svg">
          <g className="stroke-gunmetal">
            {[...Array(25)].map((_, i) => (
              <line key={`v-${i}`} x1={25.3384 + (i * 68)} y1={0} x2={25.3384 + (i * 68)} y2={1010} strokeWidth="0.673286" />
            ))}
            {[...Array(15)].map((_, i) => (
              <line key={`h-${i}`} x1={1514} y1={34.3366 + (i * 68)} x2={0} y2={34.3367 + (i * 68)} strokeWidth="0.673286" />
            ))}
          </g>
        </svg>
      </motion.div>

      <div className="container relative z-10 mx-auto px-4 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="flex flex-col gap-6 max-w-2xl"
          >
            {/* Fluid Typography with Staggered Entrance */}
            <h1 className="typef-heading-48-120 relative z-10 tracking-tight flex flex-col uppercase">
              <motion.span variants={itemVariants} className="block text-silver">When the answer is wrong,</motion.span>
              <motion.span variants={itemVariants} className="block text-white">we tell you why.</motion.span>
            </h1>
            
            <motion.p variants={itemVariants} className="type-body-1440 lg:text-xl text-silver font-medium leading-relaxed max-w-xl">
              FaultTrace-RAG doesn't just answer corpus-level questions — it proves which stage (Retrieval, Extraction, or Aggregation) caused the error, and by how much.
            </motion.p>

            <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-start sm:items-center gap-6 mt-4">
              <Link
                href="/demo"
                className="group relative inline-flex shrink-0 items-center overflow-hidden uppercase bg-neon text-black type-mono-1240 px-6 py-3.5 gap-x-4 transition-colors duration-300 before:absolute before:inset-0 before:w-0 before:bg-white before:transition-[width] before:duration-300 hover:before:w-full hover:text-black rounded-sm shadow-[0_0_20px_rgba(255,255,255,0.3)] hover:shadow-[0_0_30px_rgba(255,255,255,0.6)]"
              >
                <span className="relative z-10 transition-colors group-hover:text-black font-bold">Try the Live Trace</span>
                <div className="inline-block bg-black h-1.5 w-1.5 relative z-10 transition-colors duration-300 group-hover:bg-black" aria-hidden="true" />
              </Link>

              <Link
                href="#method"
                className="group inline-flex shrink-0 items-center overflow-hidden whitespace-nowrap outline-none transition-colors duration-300 ease-in-out has-[>svg]:gap-1.5 [&_svg]:size-3 type-mono-1240 uppercase text-silver hover:text-white"
              >
                <div className="flex items-center gap-x-2.5">[<span>Read the Protocol</span><ArrowRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />]</div>
              </Link>
            </motion.div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, scale: 0.95, rotateY: 15 }}
            animate={{ opacity: 1, scale: 1, rotateY: 0 }}
            transition={{ duration: 1, delay: 0.4, ease: [0.25, 0.4, 0.25, 1] }}
            className="relative h-[400px] w-full rounded-xl border border-gunmetal bg-surface-1/50 backdrop-blur-sm p-8 flex items-center justify-center hover:border-white/20 transition-colors duration-500 overflow-hidden shadow-2xl perspective-1000"
          >
            {/* Ambient Backlight */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-neon/10 rounded-full blur-[100px] pointer-events-none" />
            
            {/* Animated Pipeline Diagram */}
            <motion.div 
              whileHover={{ y: -10 }}
              className="relative w-full max-w-md mx-auto transition-transform duration-500"
            >
              <svg viewBox="0 0 400 150" className="w-full h-auto overflow-visible drop-shadow-2xl">
                {/* Edges */}
                <path d="M 50 75 L 120 75" stroke="rgba(255,255,255,0.2)" strokeWidth="2" strokeDasharray="4 4" className="animate-[dash_1s_linear_infinite]" />
                <path d="M 160 75 L 230 75" stroke="rgba(255,255,255,0.2)" strokeWidth="2" strokeDasharray="4 4" className="animate-[dash_1s_linear_infinite]" />
                <path d="M 270 75 L 340 75" stroke="rgba(255,255,255,0.2)" strokeWidth="2" strokeDasharray="4 4" className="animate-[dash_1s_linear_infinite]" />
                
                {/* Nodes */}
                <g transform="translate(10, 55)">
                  <rect width="40" height="40" rx="6" fill="#1f2124" stroke="rgba(255,255,255,0.2)" strokeWidth="2" />
                  <text x="20" y="25" fill="white" fontSize="12" textAnchor="middle" className="type-mono">Q</text>
                </g>

                <g transform="translate(120, 55)">
                  <rect width="40" height="40" rx="6" fill="#1f2124" stroke="rgba(255,255,255,0.2)" strokeWidth="2" />
                  <text x="20" y="25" fill="white" fontSize="12" textAnchor="middle" className="type-mono">R</text>
                  
                  {/* Fault State overlay */}
                  <rect width="40" height="40" rx="6" fill="#1f2124" stroke="var(--theme-accent)" strokeWidth="2" className="animate-pulse opacity-50" />
                  
                  {/* Tooltip */}
                  <g transform="translate(-15, -45)" className="animate-fade-in opacity-90">
                    <rect width="140" height="30" rx="4" fill="var(--theme-accent)" />
                    <text x="70" y="19" fill="black" fontSize="10" textAnchor="middle" fontWeight="bold" className="type-mono">ϕ_R = 0.62 (Scope)</text>
                    <path d="M 70 30 L 65 35 L 75 35 Z" fill="var(--theme-accent)" transform="rotate(180 70 32.5)" />
                  </g>
                </g>

                <g transform="translate(230, 55)">
                  <rect width="40" height="40" rx="6" fill="#1f2124" stroke="rgba(59, 130, 246, 0.4)" strokeWidth="2" />
                  <text x="20" y="25" fill="white" fontSize="12" textAnchor="middle" className="type-mono">E</text>
                </g>

                <g transform="translate(340, 55)">
                  <rect width="40" height="40" rx="6" fill="#1f2124" stroke="rgba(59, 130, 246, 0.4)" strokeWidth="2" />
                  <text x="20" y="25" fill="white" fontSize="12" textAnchor="middle" className="type-mono">A</text>
                </g>
              </svg>
              
              <style dangerouslySetInnerHTML={{__html: `
                @keyframes dash {
                  to { stroke-dashoffset: -8; }
                }
              `}} />
            </motion.div>
            
            <div className="absolute bottom-4 left-6 text-[10px] text-silver uppercase tracking-widest type-mono">
              Live Pipeline Trace Visualization
            </div>
          </motion.div>
          
        </div>
      </div>
    </section>
  );
}
