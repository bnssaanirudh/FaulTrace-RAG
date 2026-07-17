'use client';

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { AlertCircle } from 'lucide-react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Global error caught:', error);
  }, [error]);

  return (
    <div className="flex h-screen w-full flex-col items-center justify-center bg-mesh p-4">
      <div className="flex max-w-md flex-col items-center space-y-6 rounded-lg border border-border bg-card p-8 text-center shadow-lg">
        <div className="rounded-full bg-destructive/10 p-4">
          <AlertCircle className="h-10 w-10 text-destructive" />
        </div>
        <div className="space-y-2">
          <h2 className="text-2xl font-semibold tracking-tight">Something went wrong!</h2>
          <p className="text-sm text-muted-foreground">
            An unexpected error occurred in the application. Please try again or contact support if the issue persists.
          </p>
        </div>
        <div className="flex space-x-4">
          <Button onClick={() => window.location.reload()} variant="ghost">
            Reload Page
          </Button>
          <Button onClick={() => reset()}>
            Try Again
          </Button>
        </div>
      </div>
    </div>
  );
}
