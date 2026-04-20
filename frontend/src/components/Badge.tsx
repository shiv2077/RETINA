import { ReactNode } from 'react';

type BadgeColor = 'pass' | 'alert' | 'warn' | 'kul' | 'purple' | 'default';

interface BadgeProps {
  children: ReactNode;
  color?: BadgeColor;
  dot?: boolean;
  className?: string;
}

const colorClasses: Record<BadgeColor, string> = {
  pass:    'bg-state-passSubtle text-state-pass border border-state-pass/20',
  alert:   'bg-state-alertSubtle text-state-alert border border-state-alert/20',
  warn:    'bg-state-warnSubtle text-state-warn border border-state-warn/20',
  kul:     'bg-kul-blue/10 text-kul-accent border border-kul-accent/20',
  purple:  'bg-purple-500/10 text-purple-400 border border-purple-500/20',
  default: 'bg-surface-overlay text-text-secondary border border-surface-border',
};

const dotColorClasses: Record<BadgeColor, string> = {
  pass:    'bg-state-pass',
  alert:   'bg-state-alert',
  warn:    'bg-state-warn',
  kul:     'bg-kul-accent',
  purple:  'bg-purple-400',
  default: 'bg-text-tertiary',
};

export default function Badge({
  children,
  color = 'default',
  dot = false,
  className = '',
}: BadgeProps) {
  return (
    <span
      className={[
        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium',
        colorClasses[color],
        className,
      ].join(' ')}
    >
      {dot && (
        <span
          className={['w-1.5 h-1.5 rounded-full flex-shrink-0', dotColorClasses[color]].join(' ')}
        />
      )}
      {children}
    </span>
  );
}

export { Badge };
