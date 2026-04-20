import { ReactNode } from 'react';

type Padding = 'none' | 'sm' | 'md' | 'lg';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  alert?: boolean;
  padding?: Padding;
}

const paddingClasses: Record<Padding, string> = {
  none: '',
  sm:   'p-4',
  md:   'p-6',
  lg:   'p-8',
};

export default function GlassCard({
  children,
  className = '',
  alert = false,
  padding = 'md',
}: GlassCardProps) {
  return (
    <div
      className={[
        'glass rounded-2xl',
        paddingClasses[padding],
        alert ? 'alert-pulse border-state-alert/50' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </div>
  );
}

export { GlassCard };
