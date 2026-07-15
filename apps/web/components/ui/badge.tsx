'use client';

import { cn } from '@/lib/utils';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'brand' | 'success' | 'error' | 'warning' | 'neutral' | 'violet' | 'cyan' | 'orange';
  className?: string;
}

const variantMap = {
  brand: 'bg-brand-500/15 text-brand-300 ring-1 ring-brand-500/30',
  success: 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30',
  error: 'bg-red-500/15 text-red-300 ring-1 ring-red-500/30',
  warning: 'bg-yellow-500/15 text-yellow-300 ring-1 ring-yellow-500/30',
  neutral: 'bg-white/[0.07] text-slate-400 ring-1 ring-white/10',
  violet: 'bg-violet-500/15 text-violet-300 ring-1 ring-violet-500/30',
  cyan: 'bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30',
  orange: 'bg-orange-500/15 text-orange-300 ring-1 ring-orange-500/30',
};

export function Badge({ children, variant = 'neutral', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold',
        variantMap[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
