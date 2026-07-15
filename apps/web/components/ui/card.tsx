'use client';

import { cn } from '@/lib/utils';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: 'brand' | 'gold' | 'emerald';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const glowMap = {
  brand: 'hover:border-brand-500/30 hover:shadow-glow-brand',
  gold: 'hover:border-yellow-500/30 hover:shadow-glow-gold',
  emerald: 'hover:border-emerald-500/30 hover:shadow-glow-emerald',
};

const paddingMap = {
  none: '',
  sm: 'p-4',
  md: 'p-5',
  lg: 'p-6',
};

export function Card({ children, className, hover = false, glow, padding = 'md' }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-xl border border-white/[0.07] bg-white/[0.04] backdrop-blur-sm',
        'shadow-[0_1px_3px_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.05)]',
        hover && 'transition-all duration-200',
        hover && glow && glowMap[glow],
        paddingMap[padding],
        className
      )}
    >
      {children}
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaPositive?: boolean;
  icon?: React.ReactNode;
  glow?: 'brand' | 'gold' | 'emerald';
}

export function StatCard({ label, value, delta, deltaPositive, icon, glow }: StatCardProps) {
  return (
    <Card hover glow={glow} padding="md">
      <div className="flex items-start justify-between">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
            {label}
          </span>
          <span className="text-3xl font-bold tracking-tight text-white">{value}</span>
          {delta && (
            <span
              className={cn(
                'text-xs font-medium',
                deltaPositive ? 'text-emerald-400' : 'text-red-400'
              )}
            >
              {delta}
            </span>
          )}
        </div>
        {icon && (
          <div className="rounded-lg bg-white/[0.06] p-2.5 text-slate-400">{icon}</div>
        )}
      </div>
    </Card>
  );
}
