interface AnomalyScoreBarProps {
  score: number;
  size?: 'xs' | 'sm' | 'md' | 'lg';
  showValue?: boolean;
  className?: string;
}

const heightClasses = {
  xs: 'h-0.5',
  sm: 'h-1',
  md: 'h-1.5',
  lg: 'h-2',
};

function scoreColor(score: number): string {
  if (score < 0.4)  return 'text-state-pass';
  if (score < 0.7)  return 'text-state-warn';
  return 'text-state-alert';
}

export default function AnomalyScoreBar({
  score,
  size = 'md',
  showValue = false,
  className = '',
}: AnomalyScoreBarProps) {
  const pct = Math.min(Math.max(score, 0), 1) * 100;

  return (
    <div className={['flex items-center gap-2', className].join(' ')}>
      <div
        className={[
          'relative flex-1 bg-surface-overlay rounded-full overflow-hidden',
          heightClasses[size],
        ].join(' ')}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full score-gradient transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
      {showValue && (
        <span className={['font-mono text-xs tabular-nums', scoreColor(score)].join(' ')}>
          {score.toFixed(3)}
        </span>
      )}
    </div>
  );
}

export { AnomalyScoreBar };
