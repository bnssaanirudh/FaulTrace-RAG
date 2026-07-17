'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import {
  Activity,
  BarChart3,
  Database,
  FileDown,
  Gauge,
  Globe,
  HelpCircle,
  Microscope,
  Play,
  Settings,
  ShieldCheck,
  Zap,
} from 'lucide-react';

const navSections = [
  {
    label: 'Research',
    items: [
      { href: '/',            label: 'Overview',        icon: Gauge },
      { href: '/datasets',    label: 'Datasets',        icon: Database },
      { href: '/worlds',      label: 'Corpus Worlds',   icon: Globe },
      { href: '/queries',     label: 'Query Library',   icon: HelpCircle },
    ],
  },
  {
    label: 'Execution',
    items: [
      { href: '/run-lab',     label: 'Run Lab',         icon: Zap },
      { href: '/runs',        label: 'Run History',     icon: Play },
    ],
  },
  {
    label: 'Diagnostics',
    items: [
      { href: '/oracle',        label: 'Oracle Diagnostics', icon: Microscope },
      { href: '/certificates',  label: 'Certificates',       icon: ShieldCheck },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { href: '/experiments', label: 'Experiments',     icon: BarChart3 },
      { href: '/reports',     label: 'Reports & Exports', icon: FileDown },
      { href: '/demo-gallery',label: 'Demo Gallery',    icon: Globe },
    ],
  },
  {
    label: 'System',
    items: [
      { href: '/settings',    label: 'System & Settings', icon: Settings },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  // Live API health poll every 30s
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch('/api/v1/health', { cache: 'no-store' });
        setApiOk(res.ok);
      } catch {
        setApiOk(false);
      }
    }
    checkHealth();
    const id = setInterval(checkHealth, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <aside className="flex h-screen w-56 flex-col border-r border-white/[0.06] bg-surface-1/80 backdrop-blur-xl">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-600 shadow-glow-brand">
          <Zap className="h-4 w-4 text-white" />
        </div>
        <div>
          <p className="text-sm font-bold tracking-tight text-white">FaultTrace</p>
          <p className="text-[10px] font-medium text-orange-400 tracking-wider">RAG Analytics</p>
        </div>
      </div>

      <div className="mx-4 h-px bg-white/[0.06]" />

      {/* Navigation */}
      <nav className="flex-1 px-3 py-3 space-y-4 overflow-y-auto scrollbar-thin" aria-label="Main navigation">
        {navSections.map(section => (
          <div key={section.label}>
            <p className="mb-1 px-2 text-[9px] font-semibold uppercase tracking-widest text-slate-600">
              {section.label}
            </p>
            <div className="space-y-0.5">
              {section.items.map(item => {
                const Icon = item.icon;
                const isActive = item.href === '/'
                  ? pathname === '/'
                  : pathname.startsWith(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    aria-current={isActive ? 'page' : undefined}
                    className={cn(
                      'flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-xs font-medium transition-all duration-150',
                      isActive
                        ? 'bg-orange-600/10 text-orange-400 ring-inset ring-1 ring-orange-500/20'
                        : 'text-slate-400 hover:bg-white/[0.05] hover:text-slate-200',
                    )}
                  >
                    <Icon className={cn('h-3.5 w-3.5 flex-shrink-0', isActive ? 'text-orange-400' : 'text-slate-500')} />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Version + health footer */}
      <div className="px-4 py-4">
        <div className="rounded-lg bg-white/[0.02] border border-white/[0.06] p-3">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  apiOk === true ? 'bg-emerald-400' : apiOk === false ? 'bg-red-400' : 'bg-slate-600'
                }`}
                title={apiOk === true ? 'API online' : apiOk === false ? 'API offline' : 'Checking…'}
              />
              <Activity className="h-3 w-3 text-slate-500" />
            </div>
            <span className="text-[10px] font-mono text-slate-500">v0.7.0</span>
          </div>
          <p className="text-[10px] text-slate-600 leading-relaxed">
            Prompt 7 — Attribution &amp; Certification UI
          </p>
          <p className="text-[9px] text-slate-700 mt-0.5">~86% complete</p>
        </div>
      </div>
    </aside>
  );
}
