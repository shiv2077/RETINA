import { CSSProperties, ReactNode } from 'react';

type Padding = 'none' | 'sm' | 'md' | 'lg';

interface CardProps {
  children: ReactNode;
  className?: string;
  padding?: Padding;
  alert?: boolean;
  style?: CSSProperties;
}

const paddingClasses: Record<Padding, string> = {
  none: '',
  sm:   'p-4',
  md:   'p-6',
  lg:   'p-8',
};

export default function Card({
  children,
  className = '',
  padding = 'md',
  alert = false,
  style,
}: CardProps) {
  return (
    <div
      style={style}
      className={[
        'card',
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

export { Card };
