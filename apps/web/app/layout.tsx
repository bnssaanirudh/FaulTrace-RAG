import type { Metadata } from 'next';
import './globals.css';
import { Sidebar } from '@/components/layout/sidebar';

export const metadata: Metadata = {
  title: {
    default: 'FaultTrace-RAG',
    template: '%s | FaultTrace-RAG',
  },
  description:
    'Counterfactual Fault Localization for Corpus-Level LLM Analytics Pipelines.',
  keywords: ['LLM', 'RAG', 'fault localization', 'analytics', 'benchmarking'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-mesh min-h-screen antialiased">
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <main className="flex-1 flex flex-col overflow-hidden relative">
            <div className="flex-1 overflow-y-auto scrollbar-thin">
              {children}
            </div>
            {/* Demo Data Footer */}
            <div className="w-full bg-amber-500/10 border-t border-amber-500/20 px-4 py-1.5 text-center">
              <p className="text-xs font-medium text-amber-500/80">
                Demo Data - For Engineering Purposes Only
              </p>
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
