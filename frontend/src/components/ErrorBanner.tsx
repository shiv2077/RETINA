import { AlertTriangle } from 'lucide-react';
import GlassCard from './GlassCard';

interface ErrorBannerProps {
  message: string;
  onRetry?: () => void;
  compact?: boolean;
}

export default function ErrorBanner({ message, onRetry, compact = false }: ErrorBannerProps) {
  if (compact) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-state-alert bg-state-alertSubtle border border-state-alert/20 rounded-full px-3 py-1">
        <AlertTriangle className="w-3 h-3" />
        {message}
      </span>
    );
  }

  return (
    <GlassCard padding="sm" className="border-l-2 border-l-state-alert">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-state-alert flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text-secondary">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2 text-xs text-kul-accent hover:underline transition-colors"
            >
              Try again
            </button>
          )}
        </div>
      </div>
    </GlassCard>
  );
}

export { ErrorBanner };
