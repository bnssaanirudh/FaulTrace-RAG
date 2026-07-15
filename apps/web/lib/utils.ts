import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number | null | undefined, decimals = 2): string {
  if (n === null || n === undefined) return '—';
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(decimals);
}

export function formatPercent(n: number | null | undefined, decimals = 1): string {
  if (n === null || n === undefined) return '—';
  return `${(n * 100).toFixed(decimals)}%`;
}

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

export function familyColor(family: string): string {
  const map: Record<string, string> = {
    count: 'text-brand-400',
    mean: 'text-emerald-400',
    proportion: 'text-violet-400',
    comparison: 'text-yellow-400',
    top_k: 'text-cyan-400',
    trend: 'text-orange-400',
  };
  return map[family.toLowerCase()] ?? 'text-slate-400';
}

export function familyBadgeClass(family: string): string {
  const map: Record<string, string> = {
    count: 'bg-brand-500/15 text-brand-300 ring-1 ring-brand-500/30',
    mean: 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30',
    proportion: 'bg-violet-500/15 text-violet-300 ring-1 ring-violet-500/30',
    comparison: 'bg-yellow-500/15 text-yellow-300 ring-1 ring-yellow-500/30',
    top_k: 'bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30',
    trend: 'bg-orange-500/15 text-orange-300 ring-1 ring-orange-500/30',
  };
  return map[family.toLowerCase()] ?? 'bg-surface-4 text-slate-400 ring-1 ring-white/10';
}

export function statusBadgeClass(status: string): string {
  const map: Record<string, string> = {
    completed: 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30',
    failed: 'bg-red-500/15 text-red-300 ring-1 ring-red-500/30',
    running: 'bg-brand-500/15 text-brand-300 ring-1 ring-brand-500/30',
    pending: 'bg-yellow-500/15 text-yellow-300 ring-1 ring-yellow-500/30',
  };
  return map[status.toLowerCase()] ?? 'bg-surface-4 text-slate-400 ring-1 ring-white/10';
}

export function truncateId(id: string, len = 8): string {
  return id.slice(0, len) + '…';
}
