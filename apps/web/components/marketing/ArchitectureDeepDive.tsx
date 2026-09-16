'use client';

import { motion } from 'framer-motion';
import { Database, Cpu, BarChart3 } from 'lucide-react';

export function ArchitectureDeepDive() {
  const steps = [
    {
      icon: <Database className="w-6 h-6 text-blue-400" />,
      title: 'Dual-Engine Evaluation',
      description: 'We run queries through both DuckDB and Pandas simultaneously to ensure 100% deterministic ground-truth verification.',
    },
    {
      icon: <Cpu className="w-6 h-6 text-purple-400" />,
      title: 'Counterfactual Swaps',
      description: 'If a pipeline fails, we swap out Retrieval, Extraction, and Aggregation components with perfect "oracles" to see which swap fixes the output.',
    },
    {
      icon: <BarChart3 className="w-6 h-6 text-green-400" />,
      title: 'Shapley Attribution',
      description: 'We calculate exact Shapley values to pinpoint the exact contribution of each component to the final failure.',
    },
  ];

  return (
    <section className="py-24 relative overflow-hidden bg-theme-bg-secondary transition-colors duration-700">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-theme-bg-start/20 via-transparent to-transparent"></div>
      <div className="max-w-6xl mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-5xl font-bold mb-4 tracking-tight">
            How FaultTrace-RAG Works
          </h2>
          <p className="text-zinc-400 text-lg md:text-xl max-w-2xl mx-auto">
            A surgical approach to LLM analytics pipelines. We don't just tell you the answer is wrong; we mathematically prove <span className="text-blue-400">why</span>.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.2 }}
              whileHover={{ y: -5 }}
              className="p-8 rounded-2xl bg-zinc-900/50 border border-zinc-800 backdrop-blur-sm relative group overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="bg-zinc-800/50 w-12 h-12 rounded-xl flex items-center justify-center mb-6 border border-zinc-700">
                {step.icon}
              </div>
              <h3 className="text-xl font-semibold mb-3 text-white">{step.title}</h3>
              <p className="text-zinc-400 leading-relaxed">
                {step.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
