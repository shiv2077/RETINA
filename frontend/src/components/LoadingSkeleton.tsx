interface LoadingSkeletonProps {
  lines?: number;
  heights?: string[];
  className?: string;
}

const defaultHeights = ['h-4', 'h-6', 'h-4'];
const defaultWidths  = ['w-1/3', 'w-1/2', 'w-2/3'];

export default function LoadingSkeleton({
  lines = 3,
  heights,
  className = '',
}: LoadingSkeletonProps) {
  const resolvedHeights = heights ?? defaultHeights;

  return (
    <div className={['flex flex-col gap-2', className].join(' ')}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className={[
            'skeleton',
            resolvedHeights[i % resolvedHeights.length],
            defaultWidths[i % defaultWidths.length],
          ].join(' ')}
        />
      ))}
    </div>
  );
}

export { LoadingSkeleton };
