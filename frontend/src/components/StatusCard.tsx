import { LucideIcon } from 'lucide-react';
import GlassCard from './GlassCard';
import LoadingSkeleton from './LoadingSkeleton';

type CardColor = 'pass' | 'alert' | 'warn' | 'kul' | 'default';

interface StatusCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  color?: CardColor;
  loading?: boolean;
}

const iconColorClasses: Record<CardColor, string> = {
  pass:    'text-state-pass',
  alert:   'text-state-alert',
  warn:    'text-state-warn',
  kul:     'text-kul-accent',
  default: 'text-text-tertiary',
};

export default function StatusCard({
  title,
  value,
  icon: Icon,
  subtitle,
  color = 'default',
  loading = false,
}: StatusCardProps) {
  return (
    <GlassCard padding="md">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
          {title}
        </span>
        <Icon className={['w-4 h-4', iconColorClasses[color]].join(' ')} />
      </div>

      {/* Value */}
      <div className="mt-3 mb-1">
        {loading ? (
          <LoadingSkeleton lines={2} heights={['h-7', 'h-3']} />
        ) : (
          <>
            <p className="font-mono text-2xl font-semibold text-text-primary tabular-nums">
              {value}
            </p>
            {subtitle && (
              <p className="text-xs text-text-tertiary mt-1">{subtitle}</p>
            )}
          </>
        )}
      </div>
    </GlassCard>
  );
}

export { StatusCard };
