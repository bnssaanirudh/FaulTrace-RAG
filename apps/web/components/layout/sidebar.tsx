'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  Activity,
  BarChart3,
  Database,
  Gauge,
  Globe,
  HelpCircle,
  Play,
  Settings,
  Zap,
} from 'lucide-react';

const navItems = [
  { href: '/', label: 'Dashboard', icon: Gauge },
  { href: '/worlds', label: 'Worlds', icon: Globe },
  { href: '/queries', label: 'Queries', icon: HelpCircle },
  { href: '/runs', label: 'Pipeline Runs', icon: Play },
  { href: '/leaderboard', label: 'Leaderboard', icon: BarChart3 },
  { href: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-60 flex-col border-r border-white/[0.06] bg-surface-1/50 backdrop-blur-xl">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 shadow-glow-brand">
          <Zap className="h-4 w-4 text-white" />
        </div>
        <div>
          <p className="text-sm font-bold tracking-tight text-white">FaultTrace</p>
          <p className="text-[10px] font-medium text-brand-400 tracking-wider">RAG Analytics</p>
        </div>
      </div>

      <div className="mx-4 mb-4 h-px bg-white/[0.06]" />

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-0.5">
        <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
          Navigation
        </p>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.href === '/'
            ? pathname === '/'
            : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-150',
                isActive
                  ? 'bg-brand-600/20 text-brand-300 ring-inset ring-1 ring-brand-500/25'
                  : 'text-slate-400 hover:bg-white/[0.05] hover:text-slate-200'
              )}
            >
              <Icon className={cn('h-4 w-4', isActive ? 'text-brand-400' : 'text-slate-500')} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Version footer */}
      <div className="px-5 py-4">
        <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-3">
          <div className="flex items-center gap-2">
            <Activity className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-xs font-medium text-slate-400">v0.2.0 — Prompt 2</span>
          </div>
          <p className="mt-1 text-[10px] text-slate-600 leading-relaxed">
            P1-P5 + Attribution active
          </p>
        </div>
      </div>
    </aside>
  );
}
