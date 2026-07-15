'use client';

import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: React.ReactNode;
}

const variantMap = {
  primary: 'bg-brand-600 hover:bg-brand-500 text-white shadow-none hover:shadow-glow-brand active:bg-brand-700 active:scale-[0.98]',
  ghost: 'border border-white/10 hover:border-white/20 hover:bg-white/[0.05] text-slate-300 hover:text-white',
  danger: 'bg-red-600/20 border border-red-500/30 text-red-300 hover:bg-red-600/30 hover:border-red-400/50',
};

const sizeMap = {
  sm: 'px-3 py-1.5 text-xs gap-1.5',
  md: 'px-4 py-2 text-sm gap-2',
  lg: 'px-5 py-2.5 text-base gap-2',
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      disabled={disabled || loading}
      className={cn(
        'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed',
        variantMap[variant],
        sizeMap[size],
        className
      )}
      {...props}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
}
