interface HeatmapOverlayProps {
  imageSrc: string;
  heatmapBase64?: string | null;
  className?: string;
  alt?: string;
}

export default function HeatmapOverlay({
  imageSrc,
  heatmapBase64,
  className = '',
  alt = 'Image',
}: HeatmapOverlayProps) {
  return (
    <div className={['relative w-full overflow-hidden rounded-xl', className].join(' ')}>
      {/* Base image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageSrc}
        alt={alt}
        className="w-full h-full object-contain block"
      />
      {/* Heatmap overlay */}
      {heatmapBase64 && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/png;base64,${heatmapBase64}`}
          alt="Anomaly heatmap"
          className="absolute inset-0 w-full h-full object-contain"
          style={{ mixBlendMode: 'multiply', opacity: 0.65 }}
        />
      )}
    </div>
  );
}

export { HeatmapOverlay };
